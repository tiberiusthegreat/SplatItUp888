from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FILTER = ROOT / "pipeline" / "filter_nonfinite_gaussians.py"
TOMBSTONE_RAW_OPACITY_MAX = np.float32(-13.815508842468262)


class FilterNonfiniteGaussianTests(unittest.TestCase):
    @staticmethod
    def make_vertices(count: int = 4) -> np.ndarray:
        names = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity"]
        names += ["rot_0", "rot_1", "rot_2", "rot_3"]
        names += ["f_dc_0", "f_dc_1", "f_dc_2"]
        names += [f"f_rest_{index}" for index in range(45)]
        vertices = np.zeros(count, dtype=np.dtype([(name, "<f4") for name in names]))
        for index, name in enumerate(names):
            vertices[name] = np.arange(count, dtype=np.float32) + index / 100.0
        vertices["rot_0"] = 1.0
        return vertices

    @staticmethod
    def write_ply(
        path: Path,
        *,
        vertices: np.ndarray | None = None,
        extra_element: bool = False,
    ) -> tuple[bytes, np.ndarray]:
        if vertices is None:
            vertices = FilterNonfiniteGaussianTests.make_vertices()
        names = list(vertices.dtype.names or ())
        lines = [
            "ply",
            "format binary_little_endian 1.0",
            "comment byte preservation test",
            f"element vertex {len(vertices)}",
        ]
        lines.extend(f"property float {name}" for name in names)
        if extra_element:
            lines.append("element face 0")
        lines.extend(("end_header", ""))
        header = "\n".join(lines).encode("ascii")
        path.write_bytes(header + vertices.tobytes())
        return header, vertices

    @staticmethod
    def run_filter(input_path: Path, output_path: Path, report_path: Path):
        return subprocess.run(
            [
                sys.executable,
                str(FILTER),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--json",
                str(report_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_raw_copies_finite_records_and_reports_nonfinite_properties(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.ply"
            output = root / "output.ply"
            report_path = root / "report.json"
            vertices = self.make_vertices()
            for row in (1, 3):
                for name in ("scale_0", "scale_1", "scale_2"):
                    vertices[name][row] = np.nan
            vertices["opacity"][1] = TOMBSTONE_RAW_OPACITY_MAX
            vertices["opacity"][3] = np.nextafter(
                TOMBSTONE_RAW_OPACITY_MAX, np.float32(-np.inf)
            )
            header, vertices = self.write_ply(source, vertices=vertices)

            result = self.run_filter(source, output, report_path)

            self.assertEqual(0, result.returncode, result.stderr)
            expected_header = header.replace(b"element vertex 4", b"element vertex 2")
            expected_payload = vertices[[0, 2]].tobytes()
            self.assertEqual(expected_header + expected_payload, output.read_bytes())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(2, report["removed_vertex_count"])
            self.assertEqual(2, report["retained_vertex_count"])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), report["input"]["sha256"])
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), report["output"]["sha256"])
            self.assertEqual(
                hashlib.sha256(expected_payload).hexdigest(), report["retained_payload_sha256"]
            )
            distribution = report["removed_nonfinite_property_distribution"]
            for name in ("scale_0", "scale_1", "scale_2"):
                self.assertEqual(2, distribution[name]["nan_values"])
                self.assertEqual(0, distribution[name]["positive_infinity_values"])
                self.assertEqual(0, distribution[name]["negative_infinity_values"])
            self.assertEqual("splatitup-filter-nonfinite-gaussians-v2", report["schema_version"])
            policy = report["removal_policy"]
            self.assertEqual("brush-transparent-all-nan-scale-tombstone-v1", policy["policy_id"])
            self.assertEqual(["scale_0", "scale_1", "scale_2"], policy["scale_properties"])
            self.assertEqual("all_nan", policy["required_scale_state"])
            self.assertTrue(policy["non_scale_properties_must_be_finite"])
            self.assertFalse(policy["infinity_allowed"])
            self.assertEqual("opacity", policy["opacity_property"])
            self.assertEqual(float(TOMBSTONE_RAW_OPACITY_MAX), policy["raw_opacity_max_inclusive"])
            self.assertEqual(1e-6, policy["alpha_max"])
            evidence = report["removal_evidence"]
            for key in (
                "candidate_nonfinite_vertex_count",
                "accepted_tombstone_vertex_count",
                "all_scales_nan_vertex_count",
                "finite_non_scale_vertex_count",
                "opacity_at_or_below_threshold_vertex_count",
            ):
                self.assertEqual(2, evidence[key])
            self.assertEqual(0, evidence["rejected_nonfinite_vertex_count"])
            self.assertEqual(0, evidence["infinity_value_count"])
            self.assertLessEqual(
                evidence["removed_raw_opacity_max"], policy["raw_opacity_max_inclusive"]
            )
            self.assertTrue(report["retained_records_raw_copied"])
            self.assertTrue(report["retained_record_order_preserved"])

    def test_unsafe_nonfinite_records_fail_closed_without_artifacts(self) -> None:
        def exact_tombstone(vertices: np.ndarray, row: int = 1) -> None:
            for name in ("scale_0", "scale_1", "scale_2"):
                vertices[name][row] = np.nan
            vertices["opacity"][row] = TOMBSTONE_RAW_OPACITY_MAX

        cases = (
            (
                "partial-scale-nan",
                False,
                {"scale_0": np.nan, "opacity": TOMBSTONE_RAW_OPACITY_MAX},
                "partial_scale_nan=1",
            ),
            (
                "scale-infinity",
                False,
                {
                    "scale_0": np.nan,
                    "scale_1": np.nan,
                    "scale_2": np.inf,
                    "opacity": TOMBSTONE_RAW_OPACITY_MAX,
                },
                "any_infinity=1",
            ),
            ("position-nan", True, {"x": np.nan}, "non_scale_nonfinite=1"),
            ("sh-infinity", True, {"f_rest_11": np.inf}, "any_infinity=1"),
            ("quaternion-nan", True, {"rot_2": np.nan}, "non_scale_nonfinite=1"),
            ("opacity-nan", True, {"opacity": np.nan}, "nonfinite_opacity=1"),
            (
                "visible-opacity",
                True,
                {
                    "opacity": np.nextafter(
                        TOMBSTONE_RAW_OPACITY_MAX, np.float32(np.inf)
                    )
                },
                "visible_opacity=1",
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, start_as_tombstone, changes, expected_reason in cases:
                with self.subTest(name=name):
                    vertices = self.make_vertices(2)
                    if start_as_tombstone:
                        exact_tombstone(vertices)
                    for property_name, value in changes.items():
                        vertices[property_name][1] = value
                    source = root / f"{name}.ply"
                    output = root / f"{name}-output.ply"
                    report = root / f"{name}-report.json"
                    self.write_ply(source, vertices=vertices)

                    result = self.run_filter(source, output, report)

                    self.assertNotEqual(0, result.returncode)
                    self.assertIn("refusing broad deletion", result.stderr)
                    self.assertIn(expected_reason, result.stderr)
                    self.assertFalse(output.exists())
                    self.assertFalse(report.exists())

    def test_refuses_existing_output_or_report_without_changing_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.ply"
            self.write_ply(source)
            for existing_name in ("output", "report"):
                output = root / f"{existing_name}.ply"
                report = root / f"{existing_name}.json"
                existing = output if existing_name == "output" else report
                existing.write_bytes(b"do-not-change")

                result = self.run_filter(source, output, report)

                self.assertNotEqual(0, result.returncode)
                self.assertEqual(b"do-not-change", existing.read_bytes())
                other = report if existing_name == "output" else output
                self.assertFalse(other.exists())

    def test_malformed_schema_and_payload_fail_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "extra-element.ply"
            self.write_ply(malformed, extra_element=True)
            truncated = root / "truncated.ply"
            self.write_ply(truncated)
            truncated.write_bytes(truncated.read_bytes()[:-1])

            for source in (malformed, truncated):
                output = root / f"{source.stem}-output.ply"
                report = root / f"{source.stem}-report.json"
                result = self.run_filter(source, output, report)
                self.assertNotEqual(0, result.returncode)
                self.assertFalse(output.exists())
                self.assertFalse(report.exists())


if __name__ == "__main__":
    unittest.main()
