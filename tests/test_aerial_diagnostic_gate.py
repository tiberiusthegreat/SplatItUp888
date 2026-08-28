import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "pipeline" / "evaluate_aerial_diagnostic_gate.py"
SPEC = importlib.util.spec_from_file_location("evaluate_aerial_diagnostic_gate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_report(close_ssim, close_psnr, control_ssim):
    images = []
    for index, (ssim, psnr) in enumerate(zip(close_ssim, close_psnr), start=1):
        images.append({"image": f"close_{index}.png", "ssim": ssim, "psnr": psnr})
    for index, ssim in enumerate(control_ssim, start=1):
        images.append({"image": f"control_{index}.png", "ssim": ssim, "psnr": 28.0})
    return {
        "evaluator_version": "test-evaluator",
        "quality_status": "measured_unrated",
        "expected_holdout_renders": len(images),
        "saved_holdout_renders": len(images),
        "images": images,
    }


class AerialDiagnosticGateTests(unittest.TestCase):
    def test_current_brush_double_extension_matches_jpg_sentinels(self):
        report = make_report([0.60] * 8, [23.0] * 8, [0.70] * 4)
        for item in report["images"]:
            item["image"] = f"{Path(item['image']).stem}.jpg.png"
        result = MODULE.evaluate_gate(
            report,
            [f"close_{index}.jpg" for index in range(1, 9)],
            [f"control_{index}.jpg" for index in range(1, 5)],
            training_steps=18000,
            growth_stop_iter=15000,
            gaussian_count=3000000,
            max_splats=6000000,
        )
        self.assertEqual("MECHANICAL_PASS__AWAITING_VISUAL_QC", result["quality_status"])

    def evaluate(self, close_ssim, close_psnr, control_ssim):
        report = make_report(close_ssim, close_psnr, control_ssim)
        return MODULE.evaluate_gate(
            report,
            [f"close_{index}" for index in range(1, len(close_ssim) + 1)],
            [f"control_{index}" for index in range(1, len(control_ssim) + 1)],
            training_steps=18000,
            growth_stop_iter=15000,
            gaussian_count=3000000,
            max_splats=6000000,
        )

    def test_passes_healthy_close_views(self):
        result = self.evaluate(
            [0.59, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67],
            [23.0] * 8,
            [0.70, 0.72, 0.74, 0.76],
        )
        self.assertEqual("MECHANICAL_PASS__AWAITING_VISUAL_QC", result["quality_status"])

    def test_fails_the_previous_burwash_quality_level(self):
        result = self.evaluate(
            [0.38, 0.40, 0.43, 0.45, 0.46, 0.47, 0.49, 0.51],
            [20.08] * 8,
            [0.70, 0.72, 0.74, 0.76],
        )
        self.assertEqual("MECHANICAL_FAIL", result["quality_status"])
        self.assertGreater(len(result["reasons"]), 0)

    def test_marks_immature_controls_inconclusive(self):
        result = self.evaluate(
            [0.50] * 8,
            [21.0] * 8,
            [0.50, 0.52, 0.54, 0.56],
        )
        self.assertEqual("INCONCLUSIVE_CONTROLS", result["quality_status"])

    def test_marks_a_failing_early_run_as_growth_active(self):
        report = make_report([0.40] * 8, [20.0] * 8, [0.70] * 4)
        result = MODULE.evaluate_gate(
            report,
            [f"close_{index}" for index in range(1, 9)],
            [f"control_{index}" for index in range(1, 5)],
            training_steps=7000,
            growth_stop_iter=15000,
            gaussian_count=3000000,
            max_splats=6000000,
        )
        self.assertEqual("INCONCLUSIVE_GROWTH_ACTIVE", result["quality_status"])

    def test_cap_bound_takes_precedence(self):
        report = make_report([0.60] * 8, [24.0] * 8, [0.70] * 4)
        result = MODULE.evaluate_gate(
            report,
            [f"close_{index}" for index in range(1, 9)],
            [f"control_{index}" for index in range(1, 5)],
            training_steps=15000,
            growth_stop_iter=15000,
            gaussian_count=5970000,
            max_splats=6000000,
        )
        self.assertEqual("INCONCLUSIVE_CAP_BOUND", result["quality_status"])

    def test_missing_sentinel_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Missing close holdout"):
            report = make_report([], [], [0.7])
            MODULE.evaluate_gate(
                report,
                ["missing"],
                ["control_1"],
                training_steps=15000,
                growth_stop_iter=15000,
                gaussian_count=3000000,
                max_splats=6000000,
            )

    def test_nan_metric_fails_closed(self):
        report = make_report([float("nan")] + [0.60] * 7, [23.0] * 8, [0.70] * 4)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            MODULE.evaluate_gate(
                report,
                [f"close_{index}" for index in range(1, 9)],
                [f"control_{index}" for index in range(1, 5)],
                training_steps=15000,
                growth_stop_iter=15000,
                gaussian_count=3000000,
                max_splats=6000000,
            )

    def test_duplicate_sentinel_fails_closed(self):
        report = make_report([0.60] * 8, [23.0] * 8, [0.70] * 4)
        with self.assertRaisesRegex(ValueError, "Duplicate close"):
            MODULE.evaluate_gate(
                report,
                ["close_1", "close_1"],
                [f"control_{index}" for index in range(1, 5)],
                training_steps=15000,
                growth_stop_iter=15000,
                gaussian_count=3000000,
                max_splats=6000000,
            )

    def test_missing_training_state_fails_closed(self):
        report = make_report([0.60] * 8, [23.0] * 8, [0.70] * 4)
        with self.assertRaisesRegex(ValueError, "Training steps"):
            MODULE.evaluate_gate(
                report,
                [f"close_{index}" for index in range(1, 9)],
                [f"control_{index}" for index in range(1, 5)],
            )

    def test_healthy_early_run_still_waits_for_growth_stop(self):
        report = make_report([0.65] * 8, [25.0] * 8, [0.70] * 4)
        result = MODULE.evaluate_gate(
            report,
            [f"close_{index}" for index in range(1, 9)],
            [f"control_{index}" for index in range(1, 5)],
            training_steps=7000,
            growth_stop_iter=15000,
            gaussian_count=3000000,
            max_splats=6000000,
        )
        self.assertEqual("INCONCLUSIVE_GROWTH_ACTIVE", result["quality_status"])

    def test_duplicate_report_stem_fails_closed(self):
        report = make_report([0.60] * 8, [23.0] * 8, [0.70] * 4)
        report["images"][1]["image"] = report["images"][0]["image"]
        with self.assertRaisesRegex(ValueError, "duplicate image stems"):
            self.evaluate_report(report)

    def test_overlapping_close_and_control_fails_closed(self):
        report = make_report([0.60] * 8, [23.0] * 8, [0.70] * 4)
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            MODULE.evaluate_gate(
                report,
                ["close_1"],
                ["close_1"],
                training_steps=15000,
                growth_stop_iter=15000,
                gaussian_count=3000000,
                max_splats=6000000,
            )

    def test_bad_report_cardinality_fails_closed(self):
        report = make_report([0.60] * 8, [23.0] * 8, [0.70] * 4)
        report["saved_holdout_renders"] -= 1
        with self.assertRaisesRegex(ValueError, "cardinality"):
            self.evaluate_report(report)

    def test_growth_stop_checkpoint_without_settling_is_inconclusive(self):
        report = make_report([0.65] * 8, [25.0] * 8, [0.70] * 4)
        result = MODULE.evaluate_gate(
            report,
            [f"close_{index}" for index in range(1, 9)],
            [f"control_{index}" for index in range(1, 5)],
            training_steps=15000,
            growth_stop_iter=15000,
            gaussian_count=3000000,
            max_splats=6000000,
        )
        self.assertEqual("INCONCLUSIVE_SETTLING", result["quality_status"])

    def test_late_topology_change_prevents_a_false_settling_pass(self):
        report = make_report([0.60] * 8, [23.0] * 8, [0.70] * 4)
        result = MODULE.evaluate_gate(
            report,
            [f"close_{index}" for index in range(1, 9)],
            [f"control_{index}" for index in range(1, 5)],
            training_steps=15000,
            growth_stop_iter=12000,
            last_topology_change_iter=14200,
            gaussian_count=3000000,
            max_splats=6000000,
        )
        self.assertEqual("INCONCLUSIVE_SETTLING", result["quality_status"])
        self.assertEqual(800, result["training_state"]["settling_steps"])
        self.assertEqual(14200, result["training_state"]["last_topology_change_iter"])

    def test_input_evidence_binds_the_exact_quality_report_and_ply(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            quality_path = root / "quality.json"
            quality_path.write_text('{"quality_status":"measured_unrated"}\n', encoding="utf-8")
            ply_path = root / "candidate.ply"
            ply_path.write_bytes(b"exact-ply-payload")
            ply_sha = hashlib.sha256(ply_path.read_bytes()).hexdigest()
            ply_report_path = root / "ply_report.json"
            ply_report = {"path": str(ply_path), "sha256": ply_sha}
            ply_report_path.write_text(json.dumps(ply_report), encoding="utf-8")
            quality_bytes = quality_path.read_bytes()
            ply_report_bytes = ply_report_path.read_bytes()

            evidence = MODULE.build_input_evidence(
                quality_path,
                quality_bytes,
                ply_report_path,
                ply_report_bytes,
                ply_report,
            )
            self.assertEqual(ply_sha, evidence["ply"]["sha256"])
            self.assertEqual(
                hashlib.sha256(quality_path.read_bytes()).hexdigest(),
                evidence["quality_report"]["sha256"],
            )

            ply_path.write_bytes(b"different-ply-payload")
            with self.assertRaisesRegex(ValueError, "does not match"):
                MODULE.build_input_evidence(
                    quality_path,
                    quality_bytes,
                    ply_report_path,
                    ply_report_bytes,
                    ply_report,
                )

    def test_input_evidence_hashes_the_same_bytes_that_were_parsed(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            quality_path = root / "quality.json"
            quality_bytes = b'{"quality_status":"measured_unrated"}\n'
            quality_path.write_bytes(quality_bytes)
            ply_path = root / "candidate.ply"
            ply_path.write_bytes(b"exact-ply-payload")
            ply_sha = hashlib.sha256(ply_path.read_bytes()).hexdigest()
            ply_report = {"path": str(ply_path), "sha256": ply_sha}
            ply_report_path = root / "ply_report.json"
            ply_report_bytes = json.dumps(ply_report).encode("utf-8")
            ply_report_path.write_bytes(ply_report_bytes)

            quality_path.write_bytes(b'{"quality_status":"changed"}\n')
            ply_report_path.write_bytes(b'{"changed":true}\n')
            evidence = MODULE.build_input_evidence(
                quality_path,
                quality_bytes,
                ply_report_path,
                ply_report_bytes,
                ply_report,
            )

            self.assertEqual(
                hashlib.sha256(quality_bytes).hexdigest(),
                evidence["quality_report"]["sha256"],
            )
            self.assertEqual(
                hashlib.sha256(ply_report_bytes).hexdigest(),
                evidence["ply_report"]["sha256"],
            )

    def test_out_of_range_metric_fails_closed(self):
        report = make_report([1.1] + [0.60] * 7, [23.0] * 8, [0.70] * 4)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.evaluate_report(report)

    def evaluate_report(self, report):
        return MODULE.evaluate_gate(
            report,
            [f"close_{index}" for index in range(1, 9)],
            [f"control_{index}" for index in range(1, 5)],
            training_steps=15000,
            growth_stop_iter=15000,
            gaussian_count=3000000,
            max_splats=6000000,
        )


if __name__ == "__main__":
    unittest.main()
