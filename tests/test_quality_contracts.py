from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MotionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selector = load_module("select_frames", ROOT / "pipeline" / "select_frames.py")

    def test_motion_distinguishes_duplicate_rotation_proxy_and_parallax(self) -> None:
        rng = np.random.default_rng(7)
        source = (rng.random((360, 640)) * 255).astype(np.uint8)
        one_homography = np.roll(source, 10, axis=1)
        two_depths = np.empty_like(source)
        two_depths[:, :320] = np.roll(source[:, :320], 5, axis=1)
        two_depths[:, 320:] = np.roll(source[:, 320:], 15, axis=1)
        minority_depth = np.roll(source, 10, axis=1)
        minority_depth[:, :96] = np.roll(source[:, :96], 24, axis=1)

        self.assertTrue(self.selector.measure_motion(source, source)["near_duplicate"])
        self.assertTrue(self.selector.measure_motion(source, one_homography)["low_parallax"])
        self.assertFalse(self.selector.measure_motion(source, two_depths)["low_parallax"])
        self.assertFalse(self.selector.measure_motion(source, minority_depth)["low_parallax"])

    @staticmethod
    def adjacent_motion(
        flow: float = 0.0, *, tracking_failed: bool = False, excessive_motion: bool = False
    ) -> dict[str, object]:
        return {
            "median_flow_normalized": flow,
            "tracking_failed": tracking_failed,
            "excessive_motion": excessive_motion,
        }

    def test_cumulative_flow_keeps_legacy_bins_when_disabled(self) -> None:
        adjacent = [self.adjacent_motion() for _ in range(12)]

        bins = self.selector.build_sampling_bins(12, 3, adjacent, 0.0)

        self.assertEqual(bins, [(0, 4), (4, 8), (8, 12)])

    def test_cumulative_flow_only_subdivides_the_fast_interval(self) -> None:
        adjacent = [self.adjacent_motion(0.001) for _ in range(12)]
        for index in (5, 6, 7):
            adjacent[index] = self.adjacent_motion(0.008)

        bins = self.selector.build_sampling_bins(12, 3, adjacent, 0.0125)

        self.assertEqual(bins, [(0, 4), (4, 6), (6, 8), (8, 12)])

    def test_cumulative_flow_treats_a_failed_track_as_one_boundary(self) -> None:
        adjacent = [self.adjacent_motion(0.001) for _ in range(12)]
        adjacent[6] = self.adjacent_motion(tracking_failed=True)

        bins = self.selector.build_sampling_bins(12, 3, adjacent, 0.0125)

        self.assertEqual(bins, [(0, 4), (4, 6), (6, 8), (8, 12)])

    @staticmethod
    def shot_edge(
        *, luminance: float = 0.01, flow: float = 0.005, residual: float = 0.001
    ) -> dict[str, object]:
        return {
            "tracked_features": 300,
            "track_fraction": 0.9,
            "median_flow_normalized": flow,
            "homography_residual_normalized": residual,
            "tracking_failed": False,
            "excessive_motion": False,
            "luminance_delta_normalized": luminance,
            "hard_cut": False,
        }

    def test_hard_cut_requires_luminance_and_noncoherent_motion(self) -> None:
        coherent_pan = self.shot_edge(luminance=0.12, flow=0.06, residual=0.001)
        motion_jump = self.shot_edge(luminance=0.12, flow=0.06, residual=0.02)

        self.assertFalse(self.selector.is_hard_cut(coherent_pan))
        self.assertTrue(self.selector.is_hard_cut(motion_jump))

        malformed_boolean = dict(motion_jump)
        malformed_boolean["tracking_failed"] = "false"
        nonfinite_metric = dict(motion_jump)
        nonfinite_metric["luminance_delta_normalized"] = float("nan")
        out_of_range_metric = dict(motion_jump)
        out_of_range_metric["track_fraction"] = 1.1
        for malformed in (malformed_boolean, nonfinite_metric, out_of_range_metric):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                self.selector.is_hard_cut(malformed)

    def test_temporal_nms_rejects_a_sustained_rapid_motion_burst(self) -> None:
        edges = [self.shot_edge() for _ in range(12)]
        for index in (4, 5, 6):
            edges[index] = self.shot_edge(luminance=0.14, flow=0.06, residual=0.04)

        self.assertEqual(self.selector.detected_cut_indices(edges), [])

    def test_subtle_cut_is_the_stable_reset_after_one_isolated_pulse(self) -> None:
        edges = [
            self.shot_edge(luminance=0.005, flow=0.0005, residual=0.0001)
            for _ in range(10)
        ]
        edges[4] = self.shot_edge(luminance=0.06, flow=0.025, residual=0.001)
        edges[5] = self.shot_edge(luminance=0.008, flow=0.0004, residual=0.0001)

        self.assertEqual(self.selector.detected_cut_indices(edges), [5])
        candidates = [f"frame_{index:06d}.jpg" for index in range(1, 11)]
        manifest = self.selector.build_shot_manifest(candidates, candidates, edges)
        self.assertEqual(manifest["shot_ids"], ["shot_01", "shot_02"])
        self.assertEqual(manifest["shots"][1]["first_candidate_index"], 6)
        self.assertEqual(manifest["hard_cuts"][0]["detection_kind"], "pulse_to_reset")

    def test_shot_manifest_binds_contiguous_ranges_to_exact_frame_names(self) -> None:
        candidates = [f"frame_{index:06d}.jpg" for index in range(1, 7)]
        selected = [candidates[index] for index in (0, 2, 3, 5)]
        edges = [self.shot_edge() for _ in candidates]
        edges[3] = self.shot_edge(luminance=0.12, flow=0.06, residual=0.02)

        manifest = self.selector.build_shot_manifest(candidates, selected, edges)

        self.assertEqual(manifest["shot_ids"], ["shot_01", "shot_02"])
        self.assertEqual(
            [
                (shot["first_candidate_index"], shot["last_candidate_index"])
                for shot in manifest["shots"]
            ],
            [(1, 3), (4, 6)],
        )
        self.assertEqual(manifest["shots"][1]["first_candidate_name"], candidates[3])
        self.assertEqual(manifest["shots"][1]["selected_frame_names"], selected[2:])

        malformed = json.loads(json.dumps(manifest))
        malformed["shots"][1]["first_candidate_index"] = 5
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.selector.validate_shot_manifest(malformed, candidates, selected)

        wrong_selection = json.loads(json.dumps(manifest))
        wrong_selection["shots"][0]["selected_frame_names"] = [candidates[0]]
        with self.assertRaisesRegex(ValueError, "selected frame names"):
            self.selector.validate_shot_manifest(wrong_selection, candidates, selected)

    def test_selector_reports_the_actual_adaptive_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            selected = root / "selected"
            source.mkdir()
            rng = np.random.default_rng(12)
            texture = (rng.random((180, 320)) * 255).astype(np.uint8)
            for index in range(12):
                frame = np.roll(texture, index * 12, axis=1)
                cv2.imwrite(str(source / f"frame_{index + 1:06d}.jpg"), frame)

            report = root / "frame_quality.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "pipeline" / "select_frames.py"),
                    "--input", str(source),
                    "--output", str(selected),
                    "--target", "3",
                    "--profile", "Object",
                    "--max-cumulative-flow", "0.0125",
                    "--records", str(report),
                    "--contact-sheet", str(root / "contact.jpg"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(report.read_text(encoding="utf-8"))
            selected_count = len(list(selected.glob("*.jpg")))
            self.assertGreater(selected_count, 3)
            self.assertEqual(payload["selected_frames"], selected_count)
            self.assertEqual(payload["adaptive_extra_frames"], selected_count - 3)
            self.assertEqual(payload["max_cumulative_flow"], 0.0125)
            manifest = payload["shot_manifest"]
            candidate_names = [f"frame_{index:06d}.jpg" for index in range(1, 13)]
            selected_names = sorted(path.name for path in selected.glob("*.jpg"))
            self.selector.validate_shot_manifest(manifest, candidate_names, selected_names)
            self.assertEqual(manifest["shot_ids"], ["shot_01"])


class ColmapImageOrderContractTests(unittest.TestCase):
    def run_validator(self, names: list[str], expected: list[str]) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "database.db"
            connection = sqlite3.connect(database)
            try:
                connection.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
                connection.executemany(
                    "INSERT INTO images(image_id, name) VALUES (?, ?)",
                    list(enumerate(names, start=1)),
                )
                connection.commit()
            finally:
                connection.close()
            image_list = root / "image_order.txt"
            image_list.write_text("\n".join(expected) + "\n", encoding="utf-8")
            return subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "pipeline" / "validate_colmap_image_order.py"),
                    "--database", str(database),
                    "--image-list", str(image_list),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_accepts_exact_image_id_order(self) -> None:
        result = self.run_validator(["frame_000001.jpg", "frame_000002.jpg"], ["frame_000001.jpg", "frame_000002.jpg"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Verified chronological COLMAP image_id order", result.stdout)

    def test_rejects_out_of_order_image_ids(self) -> None:
        result = self.run_validator(["frame_000002.jpg", "frame_000001.jpg"], ["frame_000001.jpg", "frame_000002.jpg"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("chronological image order mismatch", result.stderr)


class PlyContractTests(unittest.TestCase):
    def write_ply(self, path: Path, *, wrong_rest: bool = False, integer_x: bool = False) -> None:
        names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
        names += [f"f_rest_{index}" for index in range(45)]
        if wrong_rest:
            names[-1] = "f_rest_99"
        names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        dtype = np.dtype([(name, "<i4" if name == "x" and integer_x else "<f4") for name in names])
        vertices = np.zeros(10_000, dtype=dtype)
        vertices["rot_0"] = 1.0
        header = ["ply", "format binary_little_endian 1.0", "element vertex 10000"]
        for name in names:
            scalar = "int" if name == "x" and integer_x else "float"
            header.append(f"property {scalar} {name}")
        header.extend(("end_header", ""))
        path.write_bytes("\n".join(header).encode("ascii") + vertices.tobytes())

    def verify(self, ply: Path, report: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "pipeline" / "verify_gaussian_ply.py"), str(ply), "--json", str(report)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_exact_sh3_float32_payload_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_ply(root / "valid.ply")
            result = self.verify(root / "valid.ply", root / "report.json")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrong_sh_name_integer_field_and_truncation_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, options in (("wrong-rest", {"wrong_rest": True}), ("integer", {"integer_x": True})):
                ply = root / f"{name}.ply"
                self.write_ply(ply, **options)
                self.assertNotEqual(self.verify(ply, root / f"{name}.json").returncode, 0)
            truncated = root / "truncated.ply"
            self.write_ply(truncated)
            truncated.write_bytes(truncated.read_bytes()[:-4])
            self.assertNotEqual(self.verify(truncated, root / "truncated.json").returncode, 0)


class HoldoutContractTests(unittest.TestCase):
    def test_current_brush_double_extension_maps_to_registered_jpg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            renders = root / "renders"
            references = root / "references"
            renders.mkdir()
            references.mkdir()
            pixels = np.zeros((16, 16, 3), dtype=np.uint8)
            cv2.imwrite(str(renders / "frame_000001.jpg.png"), pixels)
            cv2.imwrite(str(references / "frame_000001.jpg"), pixels)
            registered = root / "images.txt"
            registered.write_text(
                "1 1 0 0 0 0 0 0 1 frame_000001.jpg\n\n",
                encoding="utf-8",
            )
            report = root / "report.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "pipeline" / "evaluate_holdout.py"),
                    "--renders", str(renders),
                    "--reference-images", str(references),
                    "--registered-images", str(registered),
                    "--expected", "1",
                    "--json", str(report),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual("frame_000001.jpg.png", payload["images"][0]["image"])
            self.assertEqual("frame_000001.jpg", payload["images"][0]["reference"])

    def test_unregistered_holdout_camera_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            renders = root / "renders"
            references = root / "references"
            renders.mkdir()
            references.mkdir()
            pixels = np.zeros((16, 16, 3), dtype=np.uint8)
            cv2.imwrite(str(renders / "unregistered.png"), pixels)
            cv2.imwrite(str(references / "unregistered.jpg"), pixels)
            registered = root / "images.txt"
            registered.write_text(
                "1 1 0 0 0 0 0 0 1 registered.jpg\n\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "pipeline" / "evaluate_holdout.py"),
                    "--renders", str(renders),
                    "--reference-images", str(references),
                    "--registered-images", str(registered),
                    "--expected", "1",
                    "--json", str(root / "report.json"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("absent from the cleaned COLMAP model", result.stderr)


class PoseGraphContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = load_module("validate_colmap_poses", ROOT / "pipeline" / "validate_colmap_poses.py")

    def test_short_multi_pose_island_is_excluded(self) -> None:
        centers = [np.array([float(index), 0.0, 0.0]) for index in range(20)]
        centers[8] = np.array([100.0, 0.0, 0.0])
        centers[9] = np.array([-50.0, 0.0, 0.0])
        images = {
            index: {
                "name": f"frame_{index:06d}.jpg",
                "center": center,
            }
            for index, center in enumerate(centers)
        }

        metrics = self.validator.trajectory_metrics(images)

        self.assertEqual(
            metrics["excluded_isolated_poses"],
            ["frame_000008.jpg", "frame_000009.jpg"],
        )
        self.assertEqual(metrics["validated_camera_count"], 18)
        self.assertLess(metrics["maximum_to_median_step_ratio"], 20.0)

    def test_open_route_missing_runs_scale_but_house_stays_strict(self) -> None:
        limit = self.validator.maximum_missing_run_limit

        self.assertEqual(limit("Object", 1200), 5)
        self.assertEqual(limit("House", 300), 10)
        self.assertEqual(limit("House", 1200), 10)
        self.assertEqual(limit("Walkthrough", 300), 6)
        self.assertEqual(limit("Walkthrough", 1200), 24)
        self.assertEqual(limit("Walkthrough", 2400), 24)
        self.assertEqual(limit("AerialExterior", 300), 6)
        self.assertEqual(limit("AerialExterior", 1200), 24)
        self.assertLessEqual(11, limit("Walkthrough", 1200))
        self.assertGreater(11, limit("House", 1200))

    def test_shot_manifest_requires_contiguous_unique_ranges_and_frame_names(self) -> None:
        valid = {
            "schema_version": 1,
            "kind": "stitched_video_shots",
            "candidate_frame_count": 6,
            "shots": [
                {"id": "shot_01", "first_candidate_index": 1, "last_candidate_index": 3},
                {"id": "shot_02", "first_candidate_index": 4, "last_candidate_index": 6},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "shots.json"
            manifest.write_text(json.dumps(valid), encoding="utf-8")
            _payload, order, mapping = self.validator.load_shot_manifest(
                manifest, ["frame_000001.jpg", "frame_000006.jpg"]
            )
            self.assertEqual(order, ["shot_01", "shot_02"])
            self.assertEqual(mapping["frame_000006.jpg"], "shot_02")

            invalid_payloads = []
            gap = json.loads(json.dumps(valid))
            gap["shots"][1]["first_candidate_index"] = 5
            invalid_payloads.append(gap)
            overlap = json.loads(json.dumps(valid))
            overlap["shots"][1]["first_candidate_index"] = 3
            invalid_payloads.append(overlap)
            duplicate = json.loads(json.dumps(valid))
            duplicate["shots"][1]["id"] = "shot_01"
            invalid_payloads.append(duplicate)
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    manifest.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        self.validator.load_shot_manifest(
                            manifest, ["frame_000001.jpg", "frame_000006.jpg"]
                        )

            manifest.write_text(json.dumps(valid), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.validator.load_shot_manifest(manifest, ["selected.jpg"])
            with self.assertRaises(ValueError):
                self.validator.load_shot_manifest(manifest, ["frame_000007.jpg"])

    def test_shot_aware_trajectory_ignores_only_declared_cut_jumps(self) -> None:
        images = {}
        mapping = {}
        for index in range(1, 11):
            shot_id = "shot_01" if index <= 5 else "shot_02"
            local_index = index - 1 if index <= 5 else index - 6
            offset = 0.0 if index <= 5 else 100.0
            name = f"frame_{index:06d}.jpg"
            images[index] = {
                "name": name,
                "center": np.array([offset + local_index, 0.0, 0.0]),
            }
            mapping[name] = shot_id

        metrics = self.validator.trajectory_metrics_by_shot(
            images, mapping, ["shot_01", "shot_02"]
        )
        self.assertEqual(metrics["ignored_cut_transitions"], 1)
        self.assertEqual(metrics["maximum_to_median_step_ratio"], 1.0)
        self.assertEqual(metrics["first_to_last_over_path"], None)

        images[8]["center"] = np.array([200.0, 0.0, 0.0])
        images[9]["center"] = np.array([201.0, 0.0, 0.0])
        images[10]["center"] = np.array([202.0, 0.0, 0.0])
        metrics = self.validator.trajectory_metrics_by_shot(
            images, mapping, ["shot_01", "shot_02"]
        )
        self.assertGreater(metrics["maximum_to_median_step_ratio"], 20.0)

    def test_missing_runs_registration_and_graph_are_checked_per_shot(self) -> None:
        names = [f"frame_{index:06d}.jpg" for index in range(1, 21)]
        mapping = {
            name: ("shot_01" if index <= 10 else "shot_02")
            for index, name in enumerate(names, start=1)
        }
        missing = self.validator.maximum_missing_run_by_shot(
            names[:6],
            {names[0], names[1], names[4], names[5]},
            {name: ("shot_01" if index <= 3 else "shot_02") for index, name in enumerate(names[:6], start=1)},
            ["shot_01", "shot_02"],
        )
        self.assertEqual(max(missing.values()), 1)
        self.assertEqual(
            self.validator.maximum_missing_run(
                names[:6], {names[0], names[1], names[4], names[5]}
            ),
            2,
        )

        registration = self.validator.shot_registration_metrics(
            names, set(names[:18]), mapping, ["shot_01", "shot_02"]
        )
        self.assertFalse(registration["pass"])
        self.assertEqual(registration["shots"]["shot_02"]["actual_percent"], 80.0)

        model_images = {
            index: {"name": name, "center": np.zeros(3)}
            for index, name in enumerate(names[:4], start=1)
        }
        graph_mapping = {
            names[0]: "shot_01",
            names[1]: "shot_01",
            names[2]: "shot_02",
            names[3]: "shot_02",
        }
        disconnected = self.validator.shot_graph_metrics(
            model_images, [(1, 2, 20), (3, 4, 20)], graph_mapping, ["shot_01", "shot_02"]
        )
        self.assertFalse(disconnected["pass"])
        connected = self.validator.shot_graph_metrics(
            model_images,
            [(1, 2, 20), (1, 3, 15), (1, 4, 18), (2, 3, 22), (3, 4, 20)],
            graph_mapping,
            ["shot_01", "shot_02"],
        )
        self.assertTrue(connected["pass"])
        self.assertEqual(connected["verified_cross_shot_edges"], 3)
        self.assertEqual(connected["minimum_verified_edges_per_connection"], 3)

        fragile = self.validator.shot_graph_metrics(
            model_images,
            [(1, 2, 20), (2, 3, 15), (3, 4, 20)],
            graph_mapping,
            ["shot_01", "shot_02"],
        )
        self.assertFalse(fragile["pass"])
        self.assertFalse(fragile["connections"][0]["passes_minimum"])

    def test_single_pose_island_can_bridge_normal_capture_motion(self) -> None:
        centers = [np.array([float(index), 0.0, 0.0]) for index in range(20)]
        for index in range(9, 20):
            centers[index] += np.array([4.0, 0.0, 0.0])
        centers[8] = np.array([100.0, 0.0, 0.0])
        images = {
            index: {
                "name": f"frame_{index:06d}.jpg",
                "center": center,
            }
            for index, center in enumerate(centers)
        }

        metrics = self.validator.trajectory_metrics(images)

        self.assertEqual(metrics["excluded_isolated_poses"], ["frame_000008.jpg"])
        self.assertLess(metrics["maximum_to_median_step_ratio"], 20.0)

    def test_unregistered_database_edge_cannot_prove_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            selected = root / "selected"
            model.mkdir()
            selected.mkdir()
            names = [f"frame_{index:06d}.jpg" for index in range(1, 21)]
            for name in names:
                (selected / name).touch()

            image_lines = ["# synthetic COLMAP images"]
            for image_id, name in enumerate(names[:18], start=1):
                angle = 2.0 * np.pi * (image_id - 1) / 17.0
                center = np.array([np.cos(angle), np.sin(angle), 0.0])
                tvec = -center
                image_lines.extend((f"{image_id} 1 0 0 0 {tvec[0]} {tvec[1]} {tvec[2]} 1 {name}", ""))
            (model / "images.txt").write_text("\n".join(image_lines), encoding="utf-8")

            database = root / "database.db"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
            connection.execute("CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER)")
            connection.executemany("INSERT INTO images(image_id, name) VALUES(?, ?)", enumerate(names, start=1))
            max_image_id = 2_147_483_647
            edges = [(first * max_image_id + first + 1, 25) for first in range(1, 18)]
            edges.append((1 * max_image_id + 20, 1000))
            connection.executemany("INSERT INTO two_view_geometries(pair_id, rows) VALUES(?, ?)", edges)
            connection.commit()
            connection.close()

            report = root / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "pipeline" / "validate_colmap_poses.py"),
                    "--model-text", str(model),
                    "--database", str(database),
                    "--selected-images", str(selected),
                    "--profile", "Object",
                    "--registered", "18",
                    "--points", "5000",
                    "--mean-track-length", "3",
                    "--reprojection-error", "1",
                    "--json", str(report),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(payload["quality_gates"]["model_integrity"]["pass"])
            self.assertFalse(payload["quality_gates"]["capture_loop_closure"]["pass"])
            self.assertEqual(payload["quality_gates"]["capture_loop_closure"]["verified_edges"], 0)

            mismatch = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "pipeline" / "validate_colmap_poses.py"),
                    "--model-text", str(model),
                    "--database", str(database),
                    "--selected-images", str(selected),
                    "--profile", "Object",
                    "--registered", "17",
                    "--points", "5000",
                    "--mean-track-length", "3",
                    "--reprojection-error", "1",
                    "--json", str(report),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(mismatch.returncode, 0, mismatch.stderr)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(payload["quality_gates"]["model_integrity"]["pass"])
            self.assertEqual(payload["registration_percent"], 90.0)

            shifted_lines = ["# synthetic COLMAP images"]
            shifted_names = names[1:18] + names[:1]
            for image_id, name in enumerate(shifted_names, start=1):
                angle = 2.0 * np.pi * (image_id - 1) / 17.0
                center = np.array([np.cos(angle), np.sin(angle), 0.0])
                tvec = -center
                shifted_lines.extend((f"{image_id} 1 0 0 0 {tvec[0]} {tvec[1]} {tvec[2]} 1 {name}", ""))
            (model / "images.txt").write_text("\n".join(shifted_lines), encoding="utf-8")
            mismatch = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "pipeline" / "validate_colmap_poses.py"),
                    "--model-text", str(model),
                    "--database", str(database),
                    "--selected-images", str(selected),
                    "--profile", "Object",
                    "--registered", "18",
                    "--points", "5000",
                    "--mean-track-length", "3",
                    "--reprojection-error", "1",
                    "--json", str(report),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(mismatch.returncode, 0, mismatch.stderr)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(payload["quality_gates"]["model_integrity"]["pass"])
            self.assertEqual(len(payload["quality_gates"]["model_integrity"]["model_database_id_mismatches"]), 18)

    def test_open_route_passes_walkthrough_and_aerial_but_fails_strict_house(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            selected = root / "selected"
            model.mkdir()
            selected.mkdir()
            names = [f"frame_{index:06d}.jpg" for index in range(1, 21)]
            for name in names:
                (selected / name).touch()

            image_lines = ["# synthetic open COLMAP trajectory"]
            for image_id, name in enumerate(names, start=1):
                center_x = float(image_id - 1)
                image_lines.extend((f"{image_id} 1 0 0 0 {-center_x} 0 0 1 {name}", ""))
            (model / "images.txt").write_text("\n".join(image_lines), encoding="utf-8")

            database = root / "database.db"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
            connection.execute("CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER)")
            connection.executemany("INSERT INTO images(image_id, name) VALUES(?, ?)", enumerate(names, start=1))
            max_image_id = 2_147_483_647
            edges = [(first * max_image_id + first + 1, 25) for first in range(1, 20)]
            connection.executemany("INSERT INTO two_view_geometries(pair_id, rows) VALUES(?, ?)", edges)
            connection.commit()
            connection.close()

            reports: dict[str, dict[str, object]] = {}
            for profile in ("Walkthrough", "AerialExterior", "House"):
                report = root / f"{profile.lower()}.json"
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "pipeline" / "validate_colmap_poses.py"),
                        "--model-text", str(model),
                        "--database", str(database),
                        "--selected-images", str(selected),
                        "--profile", profile,
                        "--registered", "20",
                        "--points", "20000",
                        "--mean-track-length", "3",
                        "--reprojection-error", "1",
                        "--json", str(report),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                reports[profile] = json.loads(report.read_text(encoding="utf-8"))

            walkthrough_gates = reports["Walkthrough"]["quality_gates"]
            self.assertFalse(walkthrough_gates["capture_loop_closure"]["pass"])
            self.assertFalse(walkthrough_gates["capture_loop_closure"]["blocking"])
            self.assertTrue(walkthrough_gates["overall_pass"])

            aerial_gates = reports["AerialExterior"]["quality_gates"]
            self.assertFalse(aerial_gates["capture_loop_closure"]["pass"])
            self.assertFalse(aerial_gates["capture_loop_closure"]["blocking"])
            self.assertTrue(aerial_gates["overall_pass"])

            house_gates = reports["House"]["quality_gates"]
            self.assertFalse(house_gates["capture_loop_closure"]["pass"])
            self.assertTrue(house_gates["capture_loop_closure"]["blocking"])
            self.assertFalse(house_gates["overall_pass"])

    def test_stitched_shots_pass_with_connected_graph_and_nonapplicable_flat_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            selected = root / "selected"
            model.mkdir()
            selected.mkdir()
            names = [f"frame_{index:06d}.jpg" for index in range(1, 21)]
            for name in names:
                (selected / name).touch()

            image_lines = ["# synthetic stitched COLMAP trajectory"]
            for image_id, name in enumerate(names, start=1):
                center_x = float(image_id - 1) if image_id <= 10 else float(image_id + 89)
                image_lines.extend(
                    (f"{image_id} 1 0 0 0 {-center_x} 0 0 1 {name}", "")
                )
            (model / "images.txt").write_text("\n".join(image_lines), encoding="utf-8")

            database = root / "database.db"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
            connection.execute(
                "CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER)"
            )
            connection.executemany(
                "INSERT INTO images(image_id, name) VALUES(?, ?)", enumerate(names, start=1)
            )
            max_image_id = 2_147_483_647
            edges = [(first * max_image_id + first + 1, 25) for first in range(1, 20)]
            edges.extend(
                [
                    (9 * max_image_id + 11, 25),
                    (8 * max_image_id + 12, 25),
                ]
            )
            connection.executemany(
                "INSERT INTO two_view_geometries(pair_id, rows) VALUES(?, ?)", edges
            )
            connection.commit()
            connection.close()

            manifest = root / "shots.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "stitched_video_shots",
                        "candidate_frame_count": 20,
                        "shots": [
                            {
                                "id": "shot_01",
                                "first_candidate_index": 1,
                                "last_candidate_index": 10,
                            },
                            {
                                "id": "shot_02",
                                "first_candidate_index": 11,
                                "last_candidate_index": 20,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = root / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "pipeline" / "validate_colmap_poses.py"),
                    "--model-text", str(model),
                    "--database", str(database),
                    "--selected-images", str(selected),
                    "--profile", "AerialExterior",
                    "--registered", "20",
                    "--points", "20000",
                    "--mean-track-length", "3",
                    "--reprojection-error", "1",
                    "--shot-manifest", str(manifest),
                    "--json", str(report),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(report.read_text(encoding="utf-8"))
            gates = payload["quality_gates"]
            self.assertEqual(payload["gate_version"], "2.9-shot-edge-redundancy")
            self.assertTrue(gates["trajectory_continuity"]["pass"])
            self.assertTrue(gates["per_shot_registration"]["pass"])
            self.assertTrue(gates["per_shot_registration"].get("blocking", True))
            self.assertTrue(gates["shot_match_graph"]["pass"], gates["shot_match_graph"])
            self.assertTrue(gates["shot_match_graph"].get("blocking", True))
            self.assertFalse(gates["capture_loop_closure"]["applicable"])
            self.assertFalse(gates["capture_loop_closure"]["blocking"])
            self.assertTrue(gates["overall_pass"])


class ProfileSurfaceContractTests(unittest.TestCase):
    def test_all_profiles_are_exposed_without_an_object_default(self) -> None:
        runner = (ROOT / "pipeline" / "run_video_to_splat.ps1").read_text(encoding="utf-8")
        selector = (ROOT / "pipeline" / "select_frames.py").read_text(encoding="utf-8")
        validator = (ROOT / "pipeline" / "validate_colmap_poses.py").read_text(encoding="utf-8")
        blender = (ROOT / "pipeline" / "blender_handoff.py").read_text(encoding="utf-8")
        gui = (ROOT / "SplatItUp888.ps1").read_text(encoding="utf-8")

        for surface in (runner, selector, validator, blender):
            for profile in ("Object", "Walkthrough", "House", "AerialExterior"):
                self.assertIn(f'"{profile}"', surface)
        self.assertNotIn('[string]$SceneType = "Object"', runner)
        self.assertIn('"Walkthrough - open route"', gui)
        self.assertIn('"Aerial exterior / building"', gui)
        self.assertIn('"Auto - inspect and pilot"', gui)
        self.assertIn('5 { return "AerialExterior" }', gui)
        self.assertIn("default { return $null }", gui)

    def test_aerial_uses_chronological_matching_with_global_mapping(self) -> None:
        runner = (ROOT / "pipeline" / "run_video_to_splat.ps1").read_text(encoding="utf-8")

        self.assertIn(
            '$UsesChronologicalMatching = $SceneType -in @("Walkthrough", "House", "AerialExterior")',
            runner,
        )
        self.assertIn(
            '$UsesIncrementalMapper = $SceneType -in @("Walkthrough", "House")',
            runner,
        )
        self.assertIn(
            '$SolveDatabasePath = if ($UsesIncrementalMapper) { $DatabasePath } else { $CalibratedDatabasePath }',
            runner,
        )
        self.assertIn('if ($UsesChronologicalMatching) {', runner)
        self.assertIn('"sequential_matcher", "--database_path", $DatabasePath', runner)
        self.assertIn('if ($UsesIncrementalMapper) {', runner)
        self.assertIn('"global_mapper", "--database_path", $CalibratedDatabasePath', runner)

    def test_brush_tuning_parameters_are_explicit_and_resume_signed(self) -> None:
        runner = (ROOT / "pipeline" / "run_video_to_splat.ps1").read_text(encoding="utf-8")

        self.assertIn('[int]$BrushMaxSplats = 10000000', runner)
        self.assertIn('[double]$BrushScaleLossWeight = 1e-8', runner)
        self.assertIn('[int]$EvalSplitEvery = 10', runner)
        self.assertIn('"--max-splats", "$BrushMaxSplats"', runner)
        self.assertIn('"--scale-loss-weight", $brushScaleLossWeightText', runner)
        self.assertGreaterEqual(runner.count('brush_max_splats = $BrushMaxSplats'), 3)
        self.assertGreaterEqual(
            runner.count('brush_scale_loss_weight = $BrushScaleLossWeight'), 3
        )
        self.assertIn(
            'Test-ObjectProperty $marker "brush_max_splats"', runner
        )
        self.assertIn(
            'Test-ObjectProperty $marker "brush_scale_loss_weight"', runner
        )

    def test_adaptive_flow_is_final_only_and_resume_is_evidence_backed(self) -> None:
        runner = (ROOT / "pipeline" / "run_video_to_splat.ps1").read_text(encoding="utf-8")
        gui = (ROOT / "SplatItUp888.ps1").read_text(encoding="utf-8")

        self.assertIn("[double]$MaxCumulativeFlow = 0.0", runner)
        self.assertIn('"--max-cumulative-flow", $maxCumulativeFlowText', runner)
        self.assertIn('"--image_list_path", $ImageOrderPath', runner)
        self.assertIn('"--FeatureExtraction.num_threads", "1"', runner)
        self.assertIn('"--FeatureExtraction.gpu_index", "0"', runner)
        self.assertIn("validate_colmap_image_order.py", runner)
        self.assertIn('Test-ObjectProperty $marker "frame_quality_sha256"', runner)
        self.assertIn("$currentSelectCountValid", runner)
        self.assertIn("$currentFrameQualityValid", runner)
        self.assertIn('[string]$shotManifest.selected_image_set_sha256 -ne [string]$SelectState.hash', runner)
        self.assertIn('$candidateMultiplier = if ($finalMode.Checked) { 8 } else { 2 }', gui)
        self.assertIn('$maxCumulativeFlow = if ($finalMode.Checked) { "0.0125" } else { "0" }', gui)

    def test_aerial_final_is_authorized_by_a_mature_fail_closed_diagnostic(self) -> None:
        runner = (ROOT / "pipeline" / "run_video_to_splat.ps1").read_text(encoding="utf-8")

        self.assertIn('$AerialDiagnosticSteps = 18000', runner)
        self.assertIn('$AerialDiagnosticGrowthStopIter = 15000', runner)
        self.assertIn('$AerialDiagnosticMaxSplats = 6000000', runner)
        self.assertIn('$AerialDiagnosticMaxResolution = 1920', runner)
        self.assertIn('$AerialDiagnosticEvalSplitEvery = 10', runner)
        self.assertIn(
            '$RequiresAerialDiagnostic = $Trainer -eq "Brush" -and $SceneType -eq "AerialExterior" -and $TrainingSteps -ge 40000',
            runner,
        )
        self.assertIn('select_aerial_diagnostic_sentinels.py', runner)
        self.assertIn('evaluate_aerial_diagnostic_gate.py', runner)
        self.assertIn('exactly 8 close and 4 control holdouts', runner)
        self.assertEqual(
            runner.count('quality_status -ne "MECHANICAL_PASS__AWAITING_VISUAL_QC"'),
            2,
        )
        diagnostic = runner.index('"--total-steps", "$AerialDiagnosticSteps"')
        final_train = runner.index('$brushArguments = @(', diagnostic)
        final_gate = runner.index('$aerialFinalGateArguments = @(', final_train)
        publish = runner.index('Publish-VerifiedPly $latestPly.FullName', final_gate)
        self.assertLess(diagnostic, final_train)
        self.assertLess(final_train, final_gate)
        self.assertLess(final_gate, publish)
        for evidence in (
            'aerial_sentinel_selector_sha256',
            'aerial_gate_evaluator_sha256',
            'aerial_sentinel_manifest_sha256',
            'aerial_diagnostic_quality_report_sha256',
            'aerial_diagnostic_gate_report_sha256',
            'aerial_final_gate_report_sha256',
        ):
            self.assertGreaterEqual(runner.count(evidence), 2)
        self.assertGreaterEqual(runner.count("Assert-AerialGateInputEvidence"), 3)
        self.assertIn("ExpectedPlySha256", runner)
        self.assertIn("ExpectedQualityReportSha256", runner)
        self.assertIn(
            'The publish candidate does not match the files authorized by the aerial gate.',
            runner,
        )

    def test_aerial_authorization_binds_the_colmap_text_model_and_publish_inputs(self) -> None:
        runner = (ROOT / "pipeline" / "run_video_to_splat.ps1").read_text(encoding="utf-8")

        for evidence in (
            "cameras_text_sha256",
            "images_text_sha256",
            "points3d_text_sha256",
        ):
            self.assertGreaterEqual(runner.count(evidence), 3)
        self.assertIn('$diagnosticGatePayload.input_evidence', runner)
        self.assertIn('$finalGatePayload.input_evidence', runner)
        self.assertIn(
            '[string]$finalInput.ply.sha256 -eq [string]$Marker.final_ply_sha256',
            runner,
        )
        self.assertIn(
            '[string]$finalInput.quality_report.sha256 -eq [string]$Marker.quality_report_sha256',
            runner,
        )

    def test_revalidation_is_explicit_hash_checked_and_fail_closed(self) -> None:
        runner = (ROOT / "pipeline" / "run_video_to_splat.ps1").read_text(encoding="utf-8")

        artifact_check = runner.split("$existingSolveArtifactsMatch =", 1)[1].split(
            "$currentSolveValidationMatches =", 1
        )[0]
        current_validation_check = runner.split("$currentSolveValidationMatches =", 1)[1].split(
            "$canRevalidateModel =", 1
        )[0]
        self.assertIn("$RevalidateSolve", runner)
        self.assertIn('if ($RevalidateSolve -and $FromStage -ne "solve")', runner)
        self.assertIn("[string]$ShotManifestPath", runner)
        self.assertIn('$GateVersion = "2.9-shot-edge-redundancy"', runner)
        self.assertIn(
            '$poseArguments += @("--shot-manifest", $ResolvedShotManifestPath)', runner
        )
        self.assertIn("shot_manifest_path = $ResolvedShotManifestPath", runner)
        self.assertIn("shot_manifest_sha256 = $ShotManifestSha256", runner)
        self.assertIn("$solveMarker.model_hash -eq $currentModelHash", artifact_check)
        self.assertIn("$solveMarker.database_sha256 -eq $currentDatabaseSha", artifact_check)
        self.assertIn(
            "$solveMarker.reconstruction_report_sha256 -eq $currentReconstructionReportSha",
            artifact_check,
        )
        self.assertNotIn("$GateVersion", artifact_check)
        self.assertNotIn("$PoseValidatorSha256", artifact_check)
        self.assertNotIn("$RunnerScriptSha256", artifact_check)
        self.assertNotIn("shot_manifest_path", artifact_check)
        self.assertNotIn("shot_manifest_sha256", artifact_check)
        self.assertIn("shot_manifest_path", current_validation_check)
        self.assertIn("shot_manifest_sha256", current_validation_check)
        self.assertIn("if ($RevalidateSolve -and -not $canRevalidateModel)", runner)
        self.assertIn("COLMAP matching and mapping will not run", runner)
        self.assertIn('validation_mode = $validationMode', runner)
        self.assertIn('model_created_runner_script_sha256 = $modelCreatedRunnerScriptSha', runner)
        self.assertIn('validation_runner_script_sha256 = $RunnerScriptSha256', runner)
        self.assertIn('previous_solve_marker_sha256 = if ($reusedExistingModel)', runner)


@unittest.skipUnless(POWERSHELL, "Windows PowerShell is required for native logging behavior tests")
class LoggedCommandBehaviorTests(unittest.TestCase):
    def test_silent_success_creates_empty_log_and_nonzero_still_fails(self) -> None:
        runner = (ROOT / "pipeline" / "run_video_to_splat.ps1").read_text(encoding="utf-8")
        function_start = runner.index("function Invoke-LoggedCommand {")
        function_end = runner.index("\nfunction Read-Marker", function_start)
        function_source = runner[function_start:function_end]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            success_log = root / "silent-success.log"
            failure_log = root / "silent-failure.log"
            harness = root / "logged-command-contract.ps1"
            success_path = str(success_log).replace("'", "''")
            failure_path = str(failure_log).replace("'", "''")
            harness.write_text(
                function_source
                + f"""
$ErrorActionPreference = "Stop"
$nativePowerShell = (Get-Command powershell.exe).Source
Invoke-LoggedCommand -FilePath $nativePowerShell -ArgumentList @("-NoProfile", "-Command", "exit 0") -LogPath '{success_path}'
if (-not (Test-Path -LiteralPath '{success_path}')) {{ throw "Silent success did not create its log." }}
if ((Get-Item -LiteralPath '{success_path}').Length -ne 0) {{ throw "Silent success log was not empty." }}
$failed = $false
try {{
    Invoke-LoggedCommand -FilePath $nativePowerShell -ArgumentList @("-NoProfile", "-Command", "exit 7") -LogPath '{failure_path}'
}} catch {{
    $failed = $true
    if ($_.Exception.Message -notmatch "exit code 7") {{ throw }}
}}
if (-not $failed) {{ throw "A nonzero native exit did not fail." }}
if (-not (Test-Path -LiteralPath '{failure_path}')) {{ throw "Silent failure did not create its log." }}
if ((Get-Item -LiteralPath '{failure_path}').Length -ne 0) {{ throw "Silent failure log was not empty." }}
"LOGGED_COMMAND_BEHAVIOR_PASS"
""",
                encoding="utf-8",
            )

            result = subprocess.run(
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
            self.assertEqual(success_log.stat().st_size, 0)
            self.assertEqual(failure_log.stat().st_size, 0)
            self.assertIn("LOGGED_COMMAND_BEHAVIOR_PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
