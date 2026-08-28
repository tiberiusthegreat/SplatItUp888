from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTO_RUNNER = ROOT / "pipeline" / "run_auto_video_to_splat.ps1"
CORE_RUNNER = ROOT / "pipeline" / "run_video_to_splat.ps1"
RUNTIME_HELPER = ROOT / "pipeline" / "three_dgrut_runtime.ps1"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")


class AutoRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = AUTO_RUNNER.read_text(encoding="utf-8")

    def test_pilot_is_bounded_and_ends_before_training(self) -> None:
        self.assertIn('$PilotSelectedFrames = 180', self.runner)
        self.assertIn('$PilotCandidateMultiplier = 2', self.runner)
        self.assertIn('$PilotMaxLongSide = 1280', self.runner)
        self.assertIn('$pilotSolveParameters["ToStage"] = "solve"', self.runner)
        pilot_section = self.runner[
            self.runner.index("$pilotSelectParameters") : self.runner.index("$classifierReport =")
        ]
        self.assertNotIn('["ToStage"] = "train"', pilot_section)

    def test_embedded_shot_evidence_is_bound_to_current_artifacts(self) -> None:
        self.assertIn('$embedded = $frameQuality.shot_manifest', self.runner)
        self.assertIn('"source_sha256"', self.runner)
        self.assertIn('"selected_image_set_sha256"', self.runner)
        self.assertIn('[string]$source.source_sha256', self.runner)
        self.assertIn('[string]$selectMarker.selected_frame_hash', self.runner)
        self.assertIn('"shot_manifest.auto-signed.json"', self.runner)
        self.assertGreaterEqual(self.runner.count('["ShotManifestPath"]'), 2)

    def test_detector_classifier_and_attempt_contracts_are_pinned(self) -> None:
        self.assertIn('$AcceptedShotDetectorVersion = "1.1-temporal-nms"', self.runner)
        self.assertIn(
            '$AcceptedClassifierVersion = "1.0-fail-closed-pose-pilot"', self.runner
        )
        self.assertIn('$AttemptId = [Guid]::NewGuid().ToString("N")', self.runner)
        self.assertIn('"AUTO_PROFILE_SELECTED"', self.runner)
        self.assertIn('Assert-ClassifierDecision $decision $classifierReport', self.runner)

    def test_one_terminal_path_replaces_stale_canonical_decisions(self) -> None:
        self.assertIn(
            '$TerminalDecisionPath = Join-Path $FinalRoot "auto_decision.json"', self.runner
        )
        self.assertIn(
            '$ClassifierDecisionPath = Join-Path $PilotRoot "auto_decision.$AttemptId.classifier.json"',
            self.runner,
        )
        self.assertIn(
            'foreach ($staleDecisionPath in @($TerminalDecisionPath, $LegacyPilotDecisionPath))',
            self.runner,
        )
        self.assertNotIn('Copy-Item -LiteralPath $PilotDecisionPath', self.runner)

    def test_classifier_gate_view_uses_least_restrictive_production_limits(self) -> None:
        self.assertIn(
            '$PilotGatePolicyVersion = "1.0-least-restrictive-production-gates"',
            self.runner,
        )
        self.assertIn('$gates.sparse_points.pass = $points -ge 5000', self.runner)
        self.assertIn('$stepRatio -le 30.0', self.runner)
        self.assertIn('$closure.blocking = $false', self.runner)
        self.assertIn('final_profile_revalidation_required = $true', self.runner)

    def test_auto_presets_use_only_the_existing_profile_defaults(self) -> None:
        self.assertIn('[ValidateSet("Preview", "Final", "Custom")]', self.runner)
        self.assertIn('[string]$AutoPreset = "Custom"', self.runner)
        self.assertIn('$resolvedFrames = if ($Profile -eq "Object") { 150 } else { 300 }', self.runner)
        self.assertIn('$resolvedSteps = if ($Profile -eq "Object") { 7000 } else { 10000 }', self.runner)
        self.assertIn('$resolvedFrames = if ($Profile -eq "Object") { 300 } else { 1200 }', self.runner)
        self.assertIn('$resolvedSteps = if ($Profile -eq "Object") { 30000 } else { 40000 }', self.runner)
        self.assertIn(
            'Add-Member -NotePropertyName "resolved_run" -NotePropertyValue $ResolvedRun',
            self.runner,
        )
        self.assertIn('$RequestedTrainer = $Trainer', self.runner)
        self.assertIn('$ResolvedTrainer = if ($RequestedTrainer -eq "Spirula")', self.runner)
        self.assertIn('requested_trainer = $RequestedTrainer', self.runner)
        self.assertIn('resolved_trainer = $ResolvedTrainer', self.runner)
        self.assertGreaterEqual(self.runner.count('$MaxCumulativeFlow $ResolvedTrainer'), 2)

    def test_manual_runner_profiles_and_entrypoint_are_untouched(self) -> None:
        core = CORE_RUNNER.read_text(encoding="utf-8")
        self.assertRegex(
            core,
            re.compile(
                r'\[ValidateSet\("Object", "Walkthrough", "House", "AerialExterior"\)\]\s*'
                r'\[string\]\$SceneType'
            ),
        )
        self.assertNotIn('[string]$SceneType', self.runner)
        self.assertIn(
            '$CoreRunnerPath = Join-Path $PSScriptRoot "run_video_to_splat.ps1"',
            self.runner,
        )


@unittest.skipUnless(POWERSHELL, "Windows PowerShell is required for wrapper behavior tests")
class AutoRunnerBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.pipeline = self.project / "pipeline"
        self.pipeline.mkdir(parents=True)
        shutil.copy2(AUTO_RUNNER, self.pipeline / AUTO_RUNNER.name)
        shutil.copy2(RUNTIME_HELPER, self.pipeline / RUNTIME_HELPER.name)
        (self.pipeline / "run_video_to_splat.ps1").write_text(
            self.core_stub(), encoding="utf-8"
        )
        (self.pipeline / "auto_profile.py").write_text(
            self.classifier_stub(), encoding="utf-8"
        )
        self.video = self.root / "input.mp4"
        self.video.write_bytes(b"bounded auto runner test video")
        self.output = self.root / "runs"
        self.call_log = self.root / "core_calls.jsonl"
        python_path = str(Path(sys.executable).resolve()).replace("'", "''")
        self.config = self.root / "test.psd1"
        self.config.write_text(
            f"@{{ Tools = @{{ Python = '{python_path}' }} }}\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def core_stub() -> str:
        return textwrap.dedent(
            r'''
            [CmdletBinding()]
            param(
                [string]$VideoPath, [string]$RunName, [string]$SceneType,
                [int]$SelectedFrames, [int]$TrainingSteps, [string]$Trainer,
                [int]$CandidateMultiplier, [double]$MaxCumulativeFlow,
                [int]$MaxLongSide, [int]$TrainingMaxResolution,
                [int]$BrushMaxSplats, [double]$BrushScaleLossWeight,
                [int]$EvalSplitEvery, [string]$OutputRoot, [string]$ConfigPath,
                [int]$ViewerPort, [string]$FromStage, [string]$ToStage,
                [string]$ShotManifestPath,
                [switch]$AdaptiveExtraction, [switch]$NoAutoRotate,
                [switch]$NoBlender, [switch]$OpenBlender, [switch]$NoBrowser
            )
            $ErrorActionPreference = "Stop"
            $runRoot = Join-Path $OutputRoot $RunName
            New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
            [ordered]@{
                run_name = $RunName
                scene_type = $SceneType
                from_stage = $FromStage
                to_stage = $ToStage
                shot_manifest_path = $ShotManifestPath
                selected_frames = $SelectedFrames
                training_steps = $TrainingSteps
                trainer = $Trainer
            } | ConvertTo-Json -Compress | Add-Content -LiteralPath $env:AUTO_TEST_CALL_LOG

            function Write-Json([string]$Path, $Value) {
                New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
                $Value | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $Path -Encoding utf8
            }

            if ($ToStage -eq "select") {
                $candidateCount = $SelectedFrames * 2
                $sourceSha = ("a" * 64) -join ""
                $selectedSha = ("b" * 64) -join ""
                Write-Json (Join-Path $runRoot "source.json") ([ordered]@{ source_sha256 = $sourceSha })
                Write-Json (Join-Path $runRoot ".extract.complete.json") ([ordered]@{
                    raw_frames = $candidateCount
                })
                Write-Json (Join-Path $runRoot ".select.complete.json") ([ordered]@{
                    selected_frames = $SelectedFrames
                    selected_frame_hash = $selectedSha
                })
                Write-Json (Join-Path $runRoot "video_probe.json") ([ordered]@{
                    format = [ordered]@{ duration = "20.5" }
                })
                $shotManifest = [ordered]@{
                    schema_version = 1
                    kind = "stitched_video_shots"
                    detector_version = "1.1-temporal-nms"
                    candidate_frame_count = $candidateCount
                    candidate_frame_names_sha256 = (("c" * 64) -join "")
                    selected_frame_count = $SelectedFrames
                    selected_frame_names_sha256 = (("d" * 64) -join "")
                    shot_ids = @("shot_01")
                    hard_cuts = @()
                    shots = @([ordered]@{
                        id = "shot_01"
                        first_candidate_index = 1
                        last_candidate_index = $candidateCount
                    })
                }
                Write-Json (Join-Path $runRoot "frame_quality.json") ([ordered]@{
                    profile = $SceneType
                    source_frames = $candidateCount
                    selected_frames = $SelectedFrames
                    preflight = [ordered]@{
                        motion_pass = $true
                        exposure_pass = $true
                        overall_pass = $true
                    }
                    shot_manifest = $shotManifest
                })
                return
            }

            if ($ToStage -eq "solve") {
                $modelText = Join-Path $runRoot "recon\sparse_txt"
                New-Item -ItemType Directory -Force -Path $modelText | Out-Null
                "# current images model" | Set-Content -LiteralPath (Join-Path $modelText "images.txt") -Encoding utf8
                "# current points model" | Set-Content -LiteralPath (Join-Path $modelText "points3D.txt") -Encoding utf8
                $imagesSha = (Get-FileHash -LiteralPath (Join-Path $modelText "images.txt") -Algorithm SHA256).Hash.ToLowerInvariant()
                $pointsSha = (Get-FileHash -LiteralPath (Join-Path $modelText "points3D.txt") -Algorithm SHA256).Hash.ToLowerInvariant()
                $registered = [int][Math]::Ceiling($SelectedFrames * 0.95)
                $shotDetails = [ordered]@{
                    shot_01 = [ordered]@{ pass = $true; actual = 0; maximum = 4 }
                }
                $gates = [ordered]@{
                    overall_pass = $false
                    model_integrity = [ordered]@{ pass = $true }
                    registration = [ordered]@{ pass = $true; actual_percent = 95.0; minimum_percent = 90.0 }
                    sparse_points = [ordered]@{ pass = $false; actual = 10000; minimum = 20000 }
                    mean_track_length = [ordered]@{ pass = $true; actual = 4.0; minimum = 3.0 }
                    reprojection_error = [ordered]@{ pass = $true; actual_pixels = 0.5; maximum_pixels = 1.5 }
                    verified_match_graph = [ordered]@{ pass = $true }
                    maximum_missing_run = [ordered]@{ pass = $true; actual = 0; maximum = 4; shots = $shotDetails }
                    capture_loop_closure = [ordered]@{
                        pass = $false
                        blocking = $false
                        applicable = $true
                        verified_edges = 3
                        first_to_last_over_path = 0.10
                        maximum_first_to_last_over_path = 0.15
                        first_to_last_over_median_step = 20.0
                        maximum_first_to_last_over_median_step = 10.0
                    }
                    trajectory_continuity = [ordered]@{ pass = $true }
                    per_shot_registration = [ordered]@{ pass = $true }
                    shot_match_graph = [ordered]@{ pass = $true }
                }
                Write-Json (Join-Path $runRoot "reconstruction_report.json") ([ordered]@{
                    profile = $SceneType
                    selected_images = $SelectedFrames
                    registered_images = $registered
                    registration_percent = if ($env:AUTO_TEST_SCENARIO -eq "bad_gate_type") { "95.0" } else { 95.0 }
                    points = 10000
                    images_text_sha256 = $imagesSha
                    points3d_text_sha256 = $pointsSha
                    shot_manifest = [ordered]@{
                        schema_version = 1
                        kind = "stitched_video_shots"
                        candidate_frame_count = ($SelectedFrames * 2)
                        source_sha256 = (("a" * 64) -join "")
                        selected_image_set_sha256 = (("b" * 64) -join "")
                        shot_ids = @("shot_01")
                    }
                    quality_gates = $gates
                })
            }
            '''
        ).strip() + "\n"

    @staticmethod
    def classifier_stub() -> str:
        return textwrap.dedent(
            r'''
            import argparse
            import hashlib
            import json
            import os
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--video-probe")
            parser.add_argument("--frame-quality")
            parser.add_argument("--reconstruction-report")
            parser.add_argument("--images-text")
            parser.add_argument("--points3d-text")
            parser.add_argument("--json")
            args = parser.parse_args()
            scenario = os.environ["AUTO_TEST_SCENARIO"]
            if scenario == "no_write":
                raise SystemExit(0)

            report = json.loads(Path(args.reconstruction_report).read_text(encoding="utf-8-sig"))
            if scenario == "gate_object":
                gates = report["quality_gates"]
                policy = report["auto_pilot_gate_policy"]
                if not (
                    gates["overall_pass"] is True
                    and gates["sparse_points"] == {"pass": True, "actual": 10000, "minimum": 5000}
                    and gates["capture_loop_closure"]["pass"] is True
                    and gates["capture_loop_closure"]["maximum_first_to_last_over_median_step"] == 30.0
                    and policy["version"] == "1.0-least-restrictive-production-gates"
                    and policy["final_profile_revalidation_required"] is True
                ):
                    raise SystemExit(9)

            images_sha = hashlib.sha256(Path(args.images_text).read_bytes()).hexdigest()
            points_sha = hashlib.sha256(Path(args.points3d_text).read_bytes()).hexdigest()
            evidence = {
                "pilot_profile": report["profile"],
                "shot_count": len(report["shot_manifest"]["shot_ids"]),
                "selected_images": report["selected_images"],
                "registered_images": report["registered_images"],
                "sparse_points": report["points"],
                "model_text_sha256": {"images": images_sha, "points3d": points_sha},
                "capture_preflight_pass": True,
                "pose_gates_pass": report["quality_gates"]["overall_pass"],
                "verified_match_graph_pass": True,
                "trajectory_continuity_pass": True,
                "view_convergence": {"strong_inward_orbit": True},
            }
            decision = {
                "schema_version": 1,
                "classifier_version": "1.0-fail-closed-pose-pilot",
                "status": "PASS",
                "code": "AUTO_PROFILE_SELECTED",
                "profile": "Object",
                "reason": "Current bounded pilot selected Object.",
                "evidence": evidence,
            }
            if scenario == "stop":
                decision.update(
                    status="STOP",
                    code="AUTO_PROFILE_AMBIGUOUS",
                    profile=None,
                    reason="Current evidence is ambiguous.",
                    evidence={},
                )
            elif scenario == "bad_schema":
                decision["schema_version"] = 2
            elif scenario == "bad_version":
                decision["classifier_version"] = "unreviewed"
            elif scenario == "bad_code":
                decision["code"] = "WRONG"
            elif scenario == "missing_evidence":
                decision["evidence"].pop("model_text_sha256")
            elif scenario == "wrong_hash":
                decision["evidence"]["model_text_sha256"]["images"] = "0" * 64
            elif scenario == "walkthrough":
                decision["profile"] = "Walkthrough"

            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(decision), encoding="utf-8")
            raise SystemExit(2 if decision["status"] == "STOP" else 0)
            '''
        ).strip() + "\n"

    def run_wrapper(
        self,
        scenario: str,
        stale_pass: bool = False,
        auto_preset: str = "Custom",
        selected_frames: int = 12,
        training_steps: int = 1,
        trainer: str = "Brush",
    ) -> subprocess.CompletedProcess[str]:
        if stale_pass:
            for path in (
                self.output / "sample" / "auto_decision.json",
                self.output / "sample-auto-pilot" / "auto_decision.json",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "classifier_version": "1.0-fail-closed-pose-pilot",
                            "status": "PASS",
                            "code": "AUTO_PROFILE_SELECTED",
                            "profile": "AerialExterior",
                            "reason": "stale",
                            "evidence": {},
                        }
                    ),
                    encoding="utf-8",
                )
        environment = os.environ.copy()
        module_path = environment.pop("PSMODULEPATH", None)
        if module_path:
            environment["PSModulePath"] = module_path
        environment["AUTO_TEST_SCENARIO"] = scenario
        environment["AUTO_TEST_CALL_LOG"] = str(self.call_log)
        return subprocess.run(
            [
                str(POWERSHELL),
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.pipeline / AUTO_RUNNER.name),
                "-VideoPath",
                str(self.video),
                "-RunName",
                "sample",
                "-AutoPreset",
                auto_preset,
                "-SelectedFrames",
                str(selected_frames),
                "-TrainingSteps",
                str(training_steps),
                "-Trainer",
                trainer,
                "-OutputRoot",
                str(self.output),
                "-ConfigPath",
                str(self.config),
                "-ToStage",
                "train",
                "-NoBlender",
                "-NoBrowser",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )

    def calls(self) -> list[dict[str, object]]:
        if not self.call_log.exists():
            return []
        return [json.loads(line) for line in self.call_log.read_text(encoding="utf-8-sig").splitlines()]

    def terminal(self) -> dict[str, object]:
        return json.loads(
            (self.output / "sample" / "auto_decision.json").read_text(encoding="utf-8-sig")
        )

    def assert_no_final_call(self) -> None:
        self.assertFalse(
            [call for call in self.calls() if call["run_name"] == "sample"],
            self.calls(),
        )

    def test_stale_pass_is_not_reused_when_classifier_writes_nothing(self) -> None:
        result = self.run_wrapper("no_write", stale_pass=True)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_no_final_call()
        decision = self.terminal()
        self.assertEqual((decision["status"], decision["code"]), ("STOP", "AUTO_CLASSIFIER_FAILED"))
        self.assertNotEqual(decision["reason"], "stale")
        self.assertRegex(decision["attempt_id"], r"^[0-9a-f]{32}$")

    def test_malformed_or_unbound_pass_never_launches_final(self) -> None:
        for scenario in ("bad_schema", "bad_version", "bad_code", "missing_evidence", "wrong_hash"):
            with self.subTest(scenario=scenario):
                if self.call_log.exists():
                    self.call_log.unlink()
                result = self.run_wrapper(scenario)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assert_no_final_call()
                self.assertEqual(self.terminal()["code"], "AUTO_CLASSIFIER_FAILED")

    def test_malformed_pilot_gate_evidence_stops_before_classifier_and_final(self) -> None:
        result = self.run_wrapper("bad_gate_type")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_no_final_call()
        decision = self.terminal()
        self.assertEqual(
            (decision["status"], decision["code"]),
            ("STOP", "AUTO_PILOT_GATE_EVIDENCE_INVALID"),
        )

    def test_classifier_stop_is_preserved_at_the_one_terminal_path(self) -> None:
        result = self.run_wrapper("stop")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assert_no_final_call()
        decision = self.terminal()
        self.assertEqual(
            (decision["status"], decision["code"], decision["profile"]),
            ("STOP", "AUTO_PROFILE_AMBIGUOUS", None),
        )
        self.assertFalse((self.output / "sample-auto-pilot" / "auto_decision.json").exists())
        self.assertFalse(list((self.output / "sample-auto-pilot").glob("*.classifier.json")))

    def test_10k_point_20_step_object_pilot_launches_exact_object_final(self) -> None:
        result = self.run_wrapper(
            "gate_object",
            auto_preset="Final",
            selected_frames=1200,
            training_steps=40000,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        final_calls = [call for call in self.calls() if call["run_name"] == "sample"]
        self.assertEqual(len(final_calls), 2, self.calls())
        self.assertEqual({call["scene_type"] for call in final_calls}, {"Object"})
        self.assertEqual({call["selected_frames"] for call in final_calls}, {300})
        self.assertEqual({call["training_steps"] for call in final_calls}, {30000})
        self.assertEqual(
            {(call["from_stage"], call["to_stage"]) for call in final_calls},
            {("extract", "select"), ("solve", "train")},
        )
        decision = self.terminal()
        self.assertEqual((decision["status"], decision["profile"]), ("PASS", "Object"))
        self.assertEqual(decision["decision_scope"], "profile_selection")
        self.assertEqual(
            decision["evidence"]["resolved_run"],
            {
                "auto_preset": "Final",
                "selected_profile": "Object",
                "requested_selected_frames": 1200,
                "requested_training_steps": 40000,
                "resolved_selected_frames": 300,
                "resolved_training_steps": 30000,
                "requested_trainer": "Brush",
                "resolved_trainer": "Brush",
                "trainer_resolution_reason": (
                    "Auto uses Brush unless Spirula is explicitly selected; "
                    "3DGRUT-MCMC is not approved for automatic training."
                ),
            },
        )
        self.assertRegex(decision["binding"]["source_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(decision["binding"]["classifier_report_sha256"], r"^[0-9a-f]{64}$")
        self.assertFalse(list((self.output / "sample-auto-pilot").glob("*.classifier.json")))

    def test_preview_non_object_uses_non_object_preview_defaults(self) -> None:
        result = self.run_wrapper(
            "walkthrough",
            auto_preset="Preview",
            selected_frames=1200,
            training_steps=40000,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        final_calls = [call for call in self.calls() if call["run_name"] == "sample"]
        self.assertEqual({call["scene_type"] for call in final_calls}, {"Walkthrough"})
        self.assertEqual({call["selected_frames"] for call in final_calls}, {300})
        self.assertEqual({call["training_steps"] for call in final_calls}, {10000})
        resolved = self.terminal()["evidence"]["resolved_run"]
        self.assertEqual(
            (resolved["auto_preset"], resolved["resolved_selected_frames"], resolved["resolved_training_steps"]),
            ("Preview", 300, 10000),
        )

    def test_custom_preserves_requested_values(self) -> None:
        result = self.run_wrapper(
            "gate_object",
            auto_preset="Custom",
            selected_frames=777,
            training_steps=25000,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        final_calls = [call for call in self.calls() if call["run_name"] == "sample"]
        self.assertEqual({call["selected_frames"] for call in final_calls}, {777})
        self.assertEqual({call["training_steps"] for call in final_calls}, {25000})
        resolved = self.terminal()["evidence"]["resolved_run"]
        self.assertEqual(
            (
                resolved["auto_preset"],
                resolved["requested_selected_frames"],
                resolved["requested_training_steps"],
                resolved["resolved_selected_frames"],
                resolved["resolved_training_steps"],
            ),
            ("Custom", 777, 25000, 777, 25000),
        )

    def test_auto_forces_brush_and_records_requested_experimental_trainer(self) -> None:
        result = self.run_wrapper(
            "gate_object",
            auto_preset="Final",
            selected_frames=1200,
            training_steps=40000,
            trainer="3DGUT-MCMC",
        )
        self.assertEqual(0, result.returncode, result.stderr or result.stdout)
        final_calls = [call for call in self.calls() if call["run_name"] == "sample"]
        self.assertEqual({call["trainer"] for call in final_calls}, {"Brush"})
        resolved = self.terminal()["evidence"]["resolved_run"]
        self.assertEqual("3DGUT-MCMC", resolved["requested_trainer"])
        self.assertEqual("Brush", resolved["resolved_trainer"])
        self.assertIn("not approved", resolved["trainer_resolution_reason"])


if __name__ == "__main__":
    unittest.main()
