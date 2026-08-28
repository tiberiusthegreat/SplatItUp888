from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "SplatItUp888.ps1"


class GuiAutoRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = GUI.read_text(encoding="utf-8")
        environment = os.environ.copy()
        module_path = environment.pop("PSMODULEPATH", None)
        if module_path:
            environment["PSModulePath"] = module_path
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(GUI),
                "-SmokeTest",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        if result.returncode != 0:
            raise AssertionError(result.stderr or result.stdout)
        output_lines = [line for line in result.stdout.splitlines() if line.strip()]
        cls.smoke = json.loads(output_lines[-1])

    def test_auto_is_the_default_and_legacy_brush_remains_default(self) -> None:
        self.assertEqual("Auto", self.smoke["default_scene_type"])
        self.assertFalse(self.smoke["capture_selection_required"])
        self.assertEqual("Brush", self.smoke["default_trainer"])
        self.assertEqual(["Brush", "Spirula", "3DGUT-MCMC"], self.smoke["trainer_choices"])
        self.assertTrue(self.smoke["core_runner_found"])
        self.assertTrue(self.smoke["auto_runner_found"])
        self.assertTrue(self.smoke["batch_runner_found"])
        self.assertTrue(self.smoke["multi_video_queue_enabled"])

    def test_route_table_keeps_auto_and_manual_invocations_separate(self) -> None:
        routes = self.smoke["route_table"]
        self.assertEqual(
            {
                "mode": "Auto",
                "runner": "run_auto_video_to_splat.ps1",
                "scene_type": None,
                "emits_scene_type": False,
                "emits_auto_preset": True,
            },
            routes["auto"],
        )
        expected_manual = {
            "walkthrough": "Walkthrough",
            "house": "House",
            "object": "Object",
            "aerial_exterior": "AerialExterior",
        }
        for label, scene_type in expected_manual.items():
            with self.subTest(label=label):
                self.assertEqual("Manual", routes[label]["mode"])
                self.assertEqual("run_video_to_splat.ps1", routes[label]["runner"])
                self.assertEqual(scene_type, routes[label]["scene_type"])
                self.assertTrue(routes[label]["emits_scene_type"])
                self.assertFalse(routes[label]["emits_auto_preset"])

    def test_auto_and_manual_defaults_are_preserved(self) -> None:
        self.assertEqual(
            {
                "auto_preset": "Preview",
                "frames": 300,
                "training_steps": 10000,
                "max_long_side": 1600,
            },
            self.smoke["auto_preview"],
        )
        self.assertEqual(
            {
                "auto_preset": "Final",
                "frames": 1200,
                "training_steps": 40000,
                "max_long_side": 0,
            },
            self.smoke["auto_final"],
        )
        self.assertEqual(150, self.smoke["object_preview"]["frames"])
        self.assertEqual(7000, self.smoke["object_preview"]["training_steps"])
        self.assertEqual(300, self.smoke["object_final"]["frames"])
        self.assertEqual(30000, self.smoke["object_final"]["training_steps"])
        for profile in ("walkthrough_final", "house_final", "aerial_exterior_final"):
            self.assertEqual(1200, self.smoke[profile]["frames"])
            self.assertEqual(40000, self.smoke[profile]["training_steps"])
        self.assertEqual(
            {"auto_preset": "Custom", "frames": 777, "training_steps": 25000},
            self.smoke["auto_custom"],
        )

    def test_only_auto_route_emits_explicit_preset(self) -> None:
        self.assertIn('if ($route.mode -eq "Auto")', self.source)
        self.assertIn('$command += " -AutoPreset $autoPreset"', self.source)
        self.assertIn('function Get-AutoPreset', self.source)

    def test_auto_stop_reads_only_fresh_final_run_evidence(self) -> None:
        self.assertIn(
            '$decisionPath = Join-Path $script:RunRoot "auto_decision.json"',
            self.source,
        )
        self.assertIn(
            "$currentVersion -eq $script:AutoDecisionBaseline", self.source
        )
        self.assertIn(
            "$file.LastWriteTimeUtc -lt $script:PipelineStartedUtc", self.source
        )
        self.assertIn('$exitCode -eq 2 -and $script:ActiveRouteMode -eq "Auto"', self.source)
        self.assertIn("Stopped before training: $($decision.code) - $($decision.reason)", self.source)
        self.assertNotIn("-auto-pilot\\auto_decision.json", self.source)

    def test_experimental_trainer_policy_and_smoke_stop_are_visible(self) -> None:
        policy = self.smoke["manual_trainer_policy"]
        for key in ("object_mcmc", "walkthrough_mcmc", "house_mcmc"):
            self.assertEqual("EXPERIMENTAL_TRAINER_UNSUPPORTED", policy[key]["code"])
        self.assertIsNone(policy["aerial_exterior_mcmc"])
        self.assertIn('$route.mode -eq "Manual"', self.source)
        self.assertIn('Get-ManualTrainerStop ([string]$route.scene_type) $selectedTrainer', self.source)
        self.assertIn('$decisionPath = Join-Path $script:RunRoot "training_decision.json"', self.source)
        self.assertIn('$script:TrainingDecisionBaseline', self.source)
        self.assertIn('THREEDGRUT_SMOKE_AWAITING_VISUAL_QC', self.source)
        self.assertIn("No 7K, final, or publish stage ran", self.source)


if __name__ == "__main__":
    unittest.main()
