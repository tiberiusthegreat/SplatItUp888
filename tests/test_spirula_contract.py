from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class SpirulaHoldoutEvaluatorTests(unittest.TestCase):
    def run_evaluator(
        self,
        root: Path,
        metrics: dict[str, object],
        pairs: int = 2,
    ) -> subprocess.CompletedProcess[str]:
        run_dir = root / "run"
        run_dir.mkdir()
        for index in range(pairs):
            gt = np.full((24, 32, 3), 60 + index * 20, dtype=np.uint8)
            render = gt.copy()
            cv2.imwrite(str(run_dir / f"eval-gt-{index:05d}.png"), gt)
            cv2.imwrite(str(run_dir / f"eval-render-{index:05d}.png"), render)
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "pipeline" / "evaluate_spirula_holdout.py"),
                "--run-dir",
                str(run_dir),
                "--expected",
                str(pairs),
                "--json",
                str(root / "report.json"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_accepts_complete_native_pairs_and_writes_standard_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_evaluator(
                root,
                {
                    "psnr": [100.0, 100.0],
                    "ssim": [1.0, 1.0],
                    "avg_psnr": 100.0,
                    "avg_ssim": 1.0,
                    "num_eval_images": 2,
                    "training_time": 12.5,
                    "engine_vram": 2048.0,
                },
            )

            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual("Spirula", payload["trainer"])
            self.assertEqual("measured_unrated", payload["quality_status"])
            self.assertEqual(2, payload["expected_holdout_renders"])
            self.assertEqual(2, payload["saved_holdout_renders"])
            self.assertEqual(100.0, payload["mean_psnr"])
            self.assertEqual(1.0, payload["mean_ssim"])
            self.assertEqual(2, len(list(Path(payload["render_directory"]).glob("*.png"))))
            self.assertEqual(2, len(list(Path(payload["reference_directory"]).glob("*.png"))))
            self.assertEqual([], list((root / "run").glob("eval-render-*.png")))
            self.assertEqual([], list((root / "run").glob("eval-gt-*.png")))

    def test_rejects_missing_pair_or_nonfinite_native_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_evaluator(
                root,
                {
                    "psnr": [100.0, float("nan")],
                    "ssim": [1.0, 1.0],
                    "avg_psnr": 100.0,
                    "avg_ssim": 1.0,
                    "num_eval_images": 2,
                    "training_time": 12.5,
                    "engine_vram": 2048.0,
                },
            )
            self.assertNotEqual(0, result.returncode)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_evaluator(
                root,
                {
                    "psnr": [100.0, 100.0],
                    "ssim": [1.0, 1.0],
                    "avg_psnr": 100.0,
                    "avg_ssim": 1.0,
                    "num_eval_images": 2,
                    "training_time": 12.5,
                    "engine_vram": 2048.0,
                },
            )
            (root / "run" / "holdout_renders" / "eval-00001.png").unlink()
            result = subprocess.run(result.args, capture_output=True, text=True, check=False)
            self.assertNotEqual(0, result.returncode)


class SpirulaRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = (ROOT / "pipeline" / "run_video_to_splat.ps1").read_text(encoding="utf-8")
        cls.gui = (ROOT / "SplatItUp888.ps1").read_text(encoding="utf-8")
        cls.doctor = (ROOT / "scripts" / "doctor.ps1").read_text(encoding="utf-8")
        cls.config = (ROOT / "splatitup.config.example.psd1").read_text(encoding="utf-8")

    def test_spirula_is_a_real_selectable_external_backend(self) -> None:
        self.assertIn('ValidateSet("Brush", "Spirula", "3DGUT", "3DGUT-MCMC")', self.runner)
        self.assertIn('@("Brush", "Spirula", "3DGUT-MCMC")', self.gui)
        self.assertIn('Invoke-LoggedCommand $SpirulaPath', self.runner)
        self.assertIn('"train", "3dgs"', self.runner)
        self.assertIn('"--save-eval-images", "1"', self.runner)
        self.assertIn('"--eval-mode", "interval"', self.runner)

    def test_spirula_publish_and_downstream_stay_fail_closed(self) -> None:
        self.assertIn("evaluate_spirula_holdout.py", self.runner)
        self.assertIn('Publish-VerifiedPly $latestSpirulaPly.FullName $candidateQualityReport', self.runner)
        self.assertIn('$Trainer -notin @("Brush", "Spirula")', self.runner)
        self.assertIn("spirula_metrics_sha256", self.runner)
        self.assertIn("spirula_config_sha256", self.runner)
        self.assertIn("holdout_reference_set_sha256", self.runner)

    def test_configuration_and_diagnostics_expose_spirula(self) -> None:
        self.assertIn('Spirula = ""', self.config)
        self.assertIn('"Spirula (optional)"', self.doctor)


if __name__ == "__main__":
    unittest.main()
