from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARER = ROOT / "pipeline" / "prepare_3dgrut_smoke_dataset.py"
EVALUATOR = ROOT / "pipeline" / "evaluate_3dgrut_stage.py"
RUNTIME_HELPER = ROOT / "pipeline" / "three_dgrut_runtime.ps1"
CORE_RUNNER = ROOT / "pipeline" / "run_video_to_splat.ps1"
LOCAL_CONFIG = ROOT / "splatitup.local.psd1"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")


def run_process(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    module_path = environment.pop("PSMODULEPATH", None)
    if module_path:
        environment["PSModulePath"] = module_path
    kwargs.setdefault("env", environment)
    return subprocess.run(*args, **kwargs)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ThreeDGrutSmokeDatasetTests(unittest.TestCase):
    def run_preparer(self, root: Path, output_name: str) -> tuple[subprocess.CompletedProcess[str], dict]:
        dataset = root / "dataset"
        sparse = dataset / "sparse" / "0"
        sparse.mkdir(parents=True, exist_ok=True)
        (sparse / "cameras.bin").write_bytes(b"cameras")
        (sparse / "images.bin").write_bytes(b"images")
        points = root / "points3D.txt"
        with points.open("w", encoding="utf-8") as stream:
            stream.write("# COLMAP points\n")
            for index in range(10_020):
                stream.write(f"{index + 1} {index}.0 0 0 1 2 3 0.1\n")
        report = root / f"{output_name}.json"
        result = run_process(
            [
                sys.executable,
                str(PREPARER),
                "--source-dataset",
                str(dataset),
                "--source-points",
                str(points),
                "--output",
                str(root / output_name),
                "--max-points",
                "10000",
                "--json",
                str(report),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result, json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}

    def test_seed_is_deterministically_capped_before_smoke_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, first_report = self.run_preparer(root, "smoke-a")
            second, second_report = self.run_preparer(root, "smoke-b")
            self.assertEqual(0, first.returncode, first.stderr)
            self.assertEqual(0, second.returncode, second.stderr)
            self.assertEqual(10_020, first_report["source_points"]["count"])
            self.assertEqual(10_000, first_report["output"]["point_count"])
            self.assertEqual(
                first_report["output"]["points_sha256"],
                second_report["output"]["points_sha256"],
            )


class ThreeDGrutStageGateTests(unittest.TestCase):
    def make_stage(
        self,
        root: Path,
        *,
        stage: str,
        step: int,
        count: int,
        cap: int,
        metrics: tuple[float, float, float],
        baseline: Path | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        metrics_path = root / f"{stage}-metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "mean_psnr": metrics[0],
                    "mean_ssim": metrics[1],
                    "mean_lpips": metrics[2],
                }
            ),
            encoding="utf-8",
        )
        renders = root / f"{stage}-renders"
        renders.mkdir()
        (renders / "00000.png").write_bytes(b"png")
        checkpoint = root / f"{stage}-ckpt_last.pt"
        checkpoint.write_bytes(b"checkpoint")
        ply = root / f"{stage}.ply"
        ply.write_bytes(b"ply-payload")
        ply_report = root / f"{stage}-ply.json"
        ply_report.write_text(
            json.dumps(
                {
                    "valid_gaussian_ply": True,
                    "vertex_count": count,
                    "path": str(ply),
                    "sha256": sha256(ply),
                }
            ),
            encoding="utf-8",
        )
        report = root / f"{stage}-report.json"
        command = [
            sys.executable,
            str(EVALUATOR),
            "--stage",
            stage,
            "--trainer",
            "3DGUT-MCMC",
            "--expected-step",
            str(step),
            "--max-splats",
            str(cap),
            "--metrics",
            str(metrics_path),
            "--renders",
            str(renders),
            "--checkpoint",
            str(checkpoint),
            "--ply-report",
            str(ply_report),
            "--source-sha",
            "a" * 64,
            "--dataset-image-sha",
            "b" * 64,
            "--dataset-model-sha",
            "c" * 64,
            "--config-sha",
            "d" * 64,
            "--environment-sha",
            "e" * 64,
            "--json",
            str(report),
        ]
        if baseline:
            command.extend(("--baseline", str(baseline)))
        return run_process(command, capture_output=True, text=True, check=False), report

    def test_smoke_then_improved_diagnostic_pass_with_bound_caps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smoke, smoke_report = self.make_stage(
                root,
                stage="smoke",
                step=1000,
                count=250_000,
                cap=250_000,
                metrics=(12.0, 0.20, 0.80),
            )
            self.assertEqual(0, smoke.returncode, smoke.stderr)
            diagnostic, diagnostic_report = self.make_stage(
                root,
                stage="diagnostic",
                step=7000,
                count=1_000_000,
                cap=1_000_000,
                metrics=(15.0, 0.35, 0.60),
                baseline=smoke_report,
            )
            self.assertEqual(0, diagnostic.returncode, diagnostic.stderr)
            payload = json.loads(diagnostic_report.read_text(encoding="utf-8"))
            self.assertEqual("measured_unrated", payload["quality_status"])
            self.assertEqual("MECHANICAL_PASS__AWAITING_VISUAL_QC", payload["gate_status"])
            self.assertEqual(sha256(smoke_report), payload["baseline"]["sha256"])

    def test_over_cap_stage_is_written_rejected_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, report = self.make_stage(
                root,
                stage="smoke",
                step=1000,
                count=250_001,
                cap=250_000,
                metrics=(12.0, 0.20, 0.80),
            )
            self.assertEqual(2, result.returncode)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual("rejected", payload["quality_status"])
            self.assertEqual("MECHANICAL_FAIL", payload["gate_status"])


class ThreeDGrutPowerShellContractTests(unittest.TestCase):
    def test_runtime_helper_binds_the_complete_portable_windows_environment(self) -> None:
        helper = RUNTIME_HELPER.read_text(encoding="utf-8")
        for setting in (
            "CUDA_HOME",
            "CUDA_PATH",
            "PYTHONNOUSERSITE",
            "PYTHONUTF8",
            "PYTHONIOENCODING",
            "WARP_CACHE_PATH",
            "TORCH_EXTENSIONS_DIR",
            "TORCH_HOME",
            "TORCH_CUDA_ARCH_LIST",
            "TCNN_CUDA_ARCHITECTURES",
        ):
            self.assertIn(setting, helper)
        for tool in ("nvcc.exe", "slangc.exe", "cl.exe", "cmake.exe", "ninja.exe"):
            self.assertIn(tool, helper)
        for module in ("kaolin", "tinycudann", "ppisp", "fused_ssim"):
            self.assertIn(f'"{module}"', helper)
        self.assertIn("vcvars_ver=$vcVarsVersion", helper)
        self.assertIn("OrdinalIgnoreCase", helper)
        self.assertIn("--allow-unsupported-compiler", helper)
        self.assertIn("vgg16-397923af.pth", helper)
        self.assertIn("397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0", helper)
        self.assertIn('"python_utf8==$env:PYTHONUTF8"', helper)
        self.assertIn('"python_io_encoding==$env:PYTHONIOENCODING"', helper)
        self.assertIn("PythonUtf8 = $env:PYTHONUTF8", helper)
        self.assertIn("PythonIoEncoding = $env:PYTHONIOENCODING", helper)
        self.assertIn('"tcnn_cuda_architectures==$tcnnCudaArchitectures"', helper)

    def test_runner_binds_official_config_resolution_and_staged_caps(self) -> None:
        runner = (ROOT / "pipeline" / "run_video_to_splat.ps1").read_text(encoding="utf-8")
        self.assertIn('Join-Path "configs" $ThreeDGrutConfigName', runner)
        self.assertNotIn("-m pip freeze", runner)
        self.assertIn("importlib.metadata", runner)
        self.assertIn('"--max_image_size", "$TrainingMaxResolution"', runner)
        self.assertIn('max_image_size = $TrainingMaxResolution', runner)
        self.assertIn('$ThreeDGrutSmokeSteps = 1000', runner)
        self.assertIn('$ThreeDGrutSmokeMaxSplats = 250000', runner)
        self.assertIn('$ThreeDGrutDiagnosticSteps = 7000', runner)
        self.assertIn('$ThreeDGrutDiagnosticMaxSplats = 1000000', runner)
        self.assertIn('"strategy.add.max_n_gaussians=$MaxSplats"', runner)
        self.assertIn('"resume=$ResumeCheckpoint"', runner)
        self.assertIn("three_dgrut_mcmc_max_splats", runner)
        self.assertIn("importlib-metadata-runtime-v3-utf8", runner)
        self.assertIn('code = "THREEDGRUT_SMOKE_AWAITING_VISUAL_QC"', runner)
        self.assertIn('status = "AWAITING_VISUAL_QC"', runner)
        self.assertIn('exit 2', runner[runner.index('$smokeStage ='):])
        self.assertNotIn('-Stage "diagnostic"', runner)
        self.assertNotIn('-Stage "final"', runner)
        self.assertNotIn('quality_status = "not_measured"', runner)
        self.assertNotIn('$publishCandidate', runner)
        self.assertLess(
            runner.index("$ThreeDGrutRuntime = Initialize-ThreeDGrutRuntime"),
            runner.index("$ThreeDGrutEnvironmentSha256 = Get-ThreeDGrutEnvironmentFingerprint"),
        )

    def test_gui_and_doctor_use_the_shared_runtime_contract(self) -> None:
        gui = (ROOT / "SplatItUp888.ps1").read_text(encoding="utf-8")
        doctor = (ROOT / "scripts" / "doctor.ps1").read_text(encoding="utf-8")
        helper = RUNTIME_HELPER.read_text(encoding="utf-8")
        for source in (gui, helper):
            self.assertIn(r"configs\apps\colmap_3dgut.yaml", source)
            self.assertIn(r"configs\apps\colmap_3dgut_mcmc.yaml", source)
        for source in (gui, doctor):
            self.assertIn("three_dgrut_runtime.ps1", source)
            self.assertIn("Initialize-ThreeDGrutRuntime", source)
        self.assertIn("3DGRUT LPIPS/VGG cache", doctor)
        self.assertIn("LpipsVggSha256", doctor)
        self.assertIn("3DGRUT Python UTF-8 runtime", doctor)
        self.assertIn("PythonUtf8", doctor)
        self.assertIn("PythonIoEncoding", doctor)
        self.assertIn(
            '[Environment]::SetEnvironmentVariable("PYTHONUTF8", $previousPythonUtf8, "Process")',
            gui,
        )
        self.assertIn(
            '[Environment]::SetEnvironmentVariable("PYTHONIOENCODING", $previousPythonIoEncoding, "Process")',
            gui,
        )

    def test_auto_runner_initializes_runtime_before_pilot(self) -> None:
        auto = (ROOT / "pipeline" / "run_auto_video_to_splat.ps1").read_text(encoding="utf-8")
        self.assertIn('$ResolvedTrainer = if ($RequestedTrainer -eq "Spirula")', auto)
        self.assertIn('if ($ResolvedTrainer -notin @("Brush", "Spirula"))', auto)
        self.assertLess(auto.index("configs\\apps\\$threeDGrutConfigName"), auto.index("$pilotSelectParameters"))
        self.assertLess(
            auto.index("$threeDGrutRuntime = Initialize-ThreeDGrutRuntime"),
            auto.index("$pilotSelectParameters"),
        )
        self.assertNotIn("pip freeze", auto)

    def test_example_config_exposes_every_runtime_path(self) -> None:
        config = (ROOT / "splatitup.config.example.psd1").read_text(encoding="utf-8")
        for key in ("CudaHome", "VcVars", "VcVarsVersion", "RuntimeCacheRoot", "TorchCudaArchList"):
            self.assertIn(f"{key} =", config)


@unittest.skipUnless(POWERSHELL, "Windows PowerShell is required for routing behavior tests")
class ThreeDGrutRoutingBehaviorTests(unittest.TestCase):
    def test_mcmc_direct_downstream_stages_fail_before_source_or_runtime_work(self) -> None:
        for stage in ("blender", "view"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                missing_video = Path(directory) / "must-not-be-read.mp4"
                result = run_process(
                    [
                        POWERSHELL,
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(CORE_RUNNER),
                        "-VideoPath",
                        str(missing_video),
                        "-SceneType",
                        "AerialExterior",
                        "-Trainer",
                        "3DGUT-MCMC",
                        "-FromStage",
                        stage,
                        "-ToStage",
                        stage,
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                combined = result.stdout + result.stderr
                self.assertNotEqual(0, result.returncode)
                self.assertIn("THREEDGRUT_VISUAL_APPROVAL_REQUIRED", combined)
                self.assertNotIn("Video not found", combined)
                self.assertNotIn("3DGRUT is not configured", combined)

    def test_legacy_full_mcmc_marker_cannot_authorize_reuse_publish_or_handoff(self) -> None:
        runner_source = CORE_RUNNER.read_text(encoding="utf-8")
        evidence_start = runner_source.index("function Get-ThreeDGrutAerialEvidenceState")
        evidence_end = runner_source.index("function Get-SourceTreeHash", evidence_start)
        legacy_evidence_reader = runner_source[evidence_start:evidence_end]
        self.assertNotIn("three_dgrut_diagnostic_stage_report_path", legacy_evidence_reader)
        self.assertNotIn("three_dgrut_final_stage_report_path", legacy_evidence_reader)
        self.assertNotIn('$canReuseTraining = (Get-TrainChainState $null).valid', runner_source)
        self.assertIn('elseif ($Trainer -ne "Brush")', runner_source)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "train.complete.json"
            marker.write_text(
                json.dumps(
                    {
                        "trainer": "3DGUT-MCMC",
                        "status": "complete",
                        "final_ply_sha256": "a" * 64,
                        "quality_report_sha256": "b" * 64,
                        "three_dgrut_smoke_stage_report_path": "legacy-smoke.json",
                        "three_dgrut_smoke_stage_report_sha256": "c" * 64,
                        "three_dgrut_diagnostic_stage_report_path": "legacy-diagnostic.json",
                        "three_dgrut_diagnostic_stage_report_sha256": "d" * 64,
                        "three_dgrut_final_stage_report_path": "legacy-final.json",
                        "three_dgrut_final_stage_report_sha256": "e" * 64,
                    }
                ),
                encoding="utf-8",
            )
            runner_path = str(CORE_RUNNER).replace("'", "''")
            marker_path = str(marker).replace("'", "''")
            harness = root / "legacy-mcmc-refusal.ps1"
            harness.write_text(
                r'''
$ErrorActionPreference = "Stop"
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('__RUNNER__', [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw "Core runner did not parse." }
foreach ($name in @("Assert-ApprovedTrainerForDownstream", "Get-ThreeDGrutAerialEvidenceState", "Get-TrainChainState")) {
    $node = $ast.Find({
        param($candidate)
        $candidate -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $candidate.Name -eq $name
    }, $true)
    if (-not $node) { throw "Missing function: $name" }
    . ([scriptblock]::Create($node.Extent.Text))
}
$Trainer = "3DGUT-MCMC"
$RequiresThreeDGrutAerialStages = $true
$legacyMarker = Get-Content -LiteralPath '__MARKER__' -Raw | ConvertFrom-Json
function Read-Marker([string]$Name) {
    if ($Name -eq "train") { return $legacyMarker }
    return $null
}
$legacyEvidence = Get-ThreeDGrutAerialEvidenceState $legacyMarker
$trainState = Get-TrainChainState ([pscustomobject]@{ valid = $true })
if ($legacyEvidence.valid -or $trainState.valid) { throw "Legacy MCMC evidence was accepted." }
if ([string]$trainState.marker.trainer -ne "3DGUT-MCMC") { throw "The synthetic legacy marker was not read." }
if ([string]$trainState.approval_code -ne "THREEDGRUT_VISUAL_APPROVAL_REQUIRED") { throw "Missing stable refusal code." }
$blockedStages = @()
foreach ($stage in @("Blender handoff", "viewer handoff")) {
    try {
        Assert-ApprovedTrainerForDownstream $stage
        throw "The $stage was not blocked."
    } catch {
        if ($_.Exception.Message -notlike "THREEDGRUT_VISUAL_APPROVAL_REQUIRED:*") { throw }
        $blockedStages += $stage
    }
}
[ordered]@{
    legacy_evidence_valid = [bool]$legacyEvidence.valid
    train_reuse_valid = [bool]$trainState.valid
    blocked_stages = $blockedStages
} | ConvertTo-Json -Compress
'''.replace("__RUNNER__", runner_path).replace("__MARKER__", marker_path),
                encoding="utf-8",
            )
            result = run_process(
                [
                    POWERSHELL,
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(harness),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr or result.stdout)
            receipt = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertFalse(receipt["legacy_evidence_valid"])
            self.assertFalse(receipt["train_reuse_valid"])
            self.assertEqual(["Blender handoff", "viewer handoff"], receipt["blocked_stages"])

    def test_aerial_mcmc_pass_stops_after_one_smoke_invocation(self) -> None:
        runner = CORE_RUNNER.read_text(encoding="utf-8")
        training_start = runner.index("$threeDGrutAttemptOutput =")
        branch_start = runner.index("if ($RequiresThreeDGrutAerialStages) {", training_start)
        branch = runner[
            branch_start:
            runner.index(
                'throw "EXPERIMENTAL_TRAINER_UNSUPPORTED: No unrated 3DGRUT path may train, publish, or be reused."',
                branch_start,
            )
        ]
        self.assertEqual(1, branch.count("Invoke-ThreeDGrutTrainingStage"))
        self.assertIn('-Stage "smoke"', branch)
        self.assertIn('code = "THREEDGRUT_SMOKE_AWAITING_VISUAL_QC"', branch)
        self.assertIn("exit 2", branch)
        self.assertNotIn("diagnosticStage", branch)
        self.assertNotIn("finalStage", branch)
        self.assertNotIn("Publish-VerifiedPly", branch)

    def test_unrated_combinations_stop_before_video_or_tool_processing(self) -> None:
        runner = CORE_RUNNER
        cases = (
            ("AerialExterior", "3DGUT"),
            ("Object", "3DGUT-MCMC"),
            ("Walkthrough", "3DGUT-MCMC"),
            ("House", "3DGUT-MCMC"),
        )
        for profile, trainer in cases:
            with self.subTest(profile=profile, trainer=trainer), tempfile.TemporaryDirectory() as directory:
                missing_video = Path(directory) / "must-not-be-read.mp4"
                result = run_process(
                    [
                        str(POWERSHELL),
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(runner),
                        "-VideoPath",
                        str(missing_video),
                        "-SceneType",
                        profile,
                        "-Trainer",
                        trainer,
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                combined = result.stdout + result.stderr
                self.assertNotEqual(0, result.returncode)
                self.assertIn("EXPERIMENTAL_TRAINER_UNSUPPORTED", combined)
                self.assertNotIn("Video not found", combined)


@unittest.skipUnless(
    POWERSHELL and LOCAL_CONFIG.exists(),
    "Windows PowerShell and the local portable 3DGRUT config are required",
)
class ThreeDGrutUnicodeRuntimeTests(unittest.TestCase):
    def test_logged_camera_emoji_fails_under_cp1252_and_round_trips_after_init(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_log = root / "cp1252.log"
            utf8_log = root / "utf8.log"
            harness = root / "unicode-runtime-contract.ps1"
            helper_path = str(RUNTIME_HELPER).replace("'", "''")
            config_path = str(LOCAL_CONFIG).replace("'", "''")
            baseline_path = str(baseline_log).replace("'", "''")
            utf8_path = str(utf8_log).replace("'", "''")
            harness.write_text(
                f"""
$ErrorActionPreference = "Stop"
. '{helper_path}'
if (-not (Get-Command Import-PowerShellDataFile -ErrorAction SilentlyContinue)) {{
    Import-Module -Name (Join-Path $PSHOME "Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1") -Force
}}
$config = Import-PowerShellDataFile -LiteralPath '{config_path}'
$python = [string]$config.ThreeDGRUT.Python
$previousPythonUtf8 = [Environment]::GetEnvironmentVariable("PYTHONUTF8", "Process")
$previousPythonIoEncoding = [Environment]::GetEnvironmentVariable("PYTHONIOENCODING", "Process")
try {{
    $env:PYTHONUTF8 = "0"
    $env:PYTHONIOENCODING = "cp1252"
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {{
        & $python -c "print(chr(0x1F4F7))" 2>&1 | Tee-Object -FilePath '{baseline_path}' | Out-Null
        $baselineExitCode = $LASTEXITCODE
    }} finally {{
        $ErrorActionPreference = $previousErrorAction
    }}
    if ($baselineExitCode -eq 0) {{ throw "The cp1252 control unexpectedly encoded the camera emoji." }}

    $runtime = Initialize-ThreeDGrutRuntime -ThreeDGrutConfig $config.ThreeDGRUT
    if ($runtime.PythonUtf8 -ne "1" -or $runtime.PythonIoEncoding -ne "utf-8") {{
        throw "The runtime receipt did not bind Python UTF-8 mode."
    }}
    if ($runtime.FingerprintRows -notcontains "python_utf8==1" -or
        $runtime.FingerprintRows -notcontains "python_io_encoding==utf-8") {{
        throw "The runtime fingerprint did not bind Python UTF-8 mode."
    }}

    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {{
        & $runtime.Python -c "print(chr(0x1F4F7))" 2>&1 | Tee-Object -FilePath '{utf8_path}' | Out-Null
        $utf8ExitCode = $LASTEXITCODE
    }} finally {{
        $ErrorActionPreference = $previousErrorAction
    }}
    if ($utf8ExitCode -ne 0) {{ throw "The UTF-8 camera-emoji process failed with exit code $utf8ExitCode." }}
    $loggedEmoji = (Get-Content -LiteralPath '{utf8_path}' -Raw).Trim()
    $expectedEmoji = [char]::ConvertFromUtf32(0x1F4F7)
    if (-not [StringComparer]::Ordinal.Equals($loggedEmoji, $expectedEmoji)) {{
        throw "The logged camera emoji did not round-trip exactly."
    }}
}} finally {{
    [Environment]::SetEnvironmentVariable("PYTHONUTF8", $previousPythonUtf8, "Process")
    [Environment]::SetEnvironmentVariable("PYTHONIOENCODING", $previousPythonIoEncoding, "Process")
}}
if (-not [StringComparer]::Ordinal.Equals(
    [Environment]::GetEnvironmentVariable("PYTHONUTF8", "Process"),
    $previousPythonUtf8
) -or -not [StringComparer]::Ordinal.Equals(
    [Environment]::GetEnvironmentVariable("PYTHONIOENCODING", "Process"),
    $previousPythonIoEncoding
)) {{
    throw "The runner did not restore the caller's Python encoding environment."
}}
"UNICODE_RUNTIME_ROUNDTRIP_PASS"
""",
                encoding="utf-8",
            )
            result = run_process(
                [
                    POWERSHELL,
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(harness),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            self.assertTrue(baseline_log.exists())
            self.assertTrue(utf8_log.exists())
            self.assertIn("UNICODE_RUNTIME_ROUNDTRIP_PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
