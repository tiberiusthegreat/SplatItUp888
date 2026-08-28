from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "pipeline" / "auto_profile.py"
SPEC = importlib.util.spec_from_file_location("auto_profile", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
IMAGES_SHA256 = "1" * 64
POINTS_SHA256 = "2" * 64


def probe(duration: float) -> dict[str, object]:
    return {"format": {"duration": str(duration)}}


def frame_quality(profile: str = "AerialExterior", passed: bool = True) -> dict[str, object]:
    return {
        "profile": profile,
        "selected_frames": 180,
        "preflight": {"motion_pass": passed, "exposure_pass": True, "overall_pass": passed},
    }


def reconstruction(
    *,
    profile: str = "AerialExterior",
    loop_closed: bool = False,
    passed: bool = True,
    shots: int = 1,
    registered: int = 175,
    points: int = 25_000,
) -> dict[str, object]:
    manifest = {
        "schema_version": 1,
        "kind": "stitched_video_shots",
        "candidate_frame_count": 360,
        "shot_ids": [f"shot_{index:02d}" for index in range(1, shots + 1)],
    }
    return {
        "profile": profile,
        "selected_images": 180,
        "registered_images": registered,
        "registration_percent": round(100.0 * registered / 180, 3),
        "points": points,
        "images_text_sha256": IMAGES_SHA256,
        "points3d_text_sha256": POINTS_SHA256,
        "shot_manifest": manifest,
        "quality_gates": {
            "overall_pass": passed,
            "verified_match_graph": {"pass": passed},
            "trajectory_continuity": {"pass": passed},
            "capture_loop_closure": {
                "pass": loop_closed,
                "blocking": False,
                "applicable": shots == 1,
                "verified_edges": 3 if loop_closed else 0,
            },
        },
    }


def convergence(
    strong: bool,
    registered: int = 175,
    points: int = 25_000,
    angular_orbit: bool = True,
) -> dict[str, object]:
    return {
        "registered_views": registered,
        "sparse_points": points,
        "images_text_sha256": IMAGES_SHA256,
        "points3d_text_sha256": POINTS_SHA256,
        "median_cosine": 0.98 if strong else 0.90,
        "p10_cosine": 0.57 if strong else -0.14,
        "positive_fraction": 0.98 if strong else 0.87,
        "angular_coverage_degrees": 350.0 if angular_orbit else 0.0,
        "angular_sweep_degrees": 360.0 if angular_orbit else 0.0,
        "camera_median_radius": 5.0,
        "scene_point_p80_radius": 2.0,
        "camera_to_scene_extent_ratio": 2.5,
        "points_inside_camera_shell_fraction": 0.95,
    }


class AutoProfileDecisionTests(unittest.TestCase):
    def decide(
        self,
        *,
        duration: float,
        strong_orbit: bool,
        loop_closed: bool = False,
        shots: int = 1,
        capture_pass: bool = True,
        pose_pass: bool = True,
    ) -> dict[str, object]:
        return MODULE.decide_profile(
            probe(duration),
            frame_quality(passed=capture_pass),
            reconstruction(loop_closed=loop_closed, passed=pose_pass, shots=shots),
            convergence(strong_orbit),
        )

    def test_short_closed_inward_orbit_selects_object(self) -> None:
        decision = self.decide(duration=20.5, strong_orbit=True, loop_closed=True)
        self.assertEqual((decision["status"], decision["profile"]), ("PASS", "Object"))

    def test_single_shot_open_non_orbit_selects_walkthrough(self) -> None:
        decision = self.decide(duration=301.3, strong_orbit=False)
        self.assertEqual((decision["status"], decision["profile"]), ("PASS", "Walkthrough"))

    def test_long_multi_shot_inward_orbit_selects_aerial(self) -> None:
        decision = self.decide(duration=535.3, strong_orbit=True, shots=6)
        self.assertEqual((decision["status"], decision["profile"]), ("PASS", "AerialExterior"))

    def test_closed_non_orbit_selects_house(self) -> None:
        decision = self.decide(duration=360.0, strong_orbit=False, loop_closed=True)
        self.assertEqual((decision["status"], decision["profile"]), ("PASS", "House"))

    def test_short_open_orbit_stops_as_ambiguous(self) -> None:
        decision = self.decide(duration=60.0, strong_orbit=True)
        self.assertEqual((decision["status"], decision["code"]), ("STOP", "AUTO_PROFILE_AMBIGUOUS"))

    def test_straight_push_facing_scene_is_not_an_orbit(self) -> None:
        decision = MODULE.decide_profile(
            probe(45.0),
            frame_quality(),
            reconstruction(loop_closed=False),
            convergence(True, angular_orbit=False),
        )
        self.assertEqual((decision["status"], decision["profile"]), ("PASS", "Walkthrough"))

    def test_enclosing_room_loop_is_not_an_object_orbit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images_path = root / "images.txt"
            points_path = root / "points3D.txt"
            image_lines: list[str] = ["# cameras circle inside a larger enclosing room"]
            for index in range(1, 31):
                angle = 2.0 * math.pi * (index - 1) / 30.0
                image_lines.extend(
                    [
                        f"{index} {math.cos(angle / 2.0):.9f} 0 {math.sin(angle / 2.0):.9f} 0 0 0 3 1 frame_{index:06d}.jpg",
                        "",
                    ]
                )
            images_path.write_text("\n".join(image_lines) + "\n", encoding="utf-8")
            point_lines: list[str] = []
            for index in range(1, 61):
                angle = 2.0 * math.pi * (index - 1) / 60.0
                point_lines.append(
                    f"{index} {6.0 * math.cos(angle):.9f} 0 {6.0 * math.sin(angle):.9f} 100 100 100 0.1 1 0"
                )
            points_path.write_text("\n".join(point_lines) + "\n", encoding="utf-8")

            measured = MODULE.measure_view_convergence(images_path, points_path)
            report = reconstruction(loop_closed=True, registered=30, points=60)
            report["images_text_sha256"] = measured["images_text_sha256"]
            report["points3d_text_sha256"] = measured["points3d_text_sha256"]
            decision = MODULE.decide_profile(
                probe(45.0),
                frame_quality(),
                report,
                measured,
            )

        self.assertEqual((decision["status"], decision["code"]), ("STOP", "AUTO_PROFILE_AMBIGUOUS"))
        self.assertFalse(decision["evidence"]["view_convergence"]["cameras_outside_compact_subject"])

    def test_multi_shot_non_orbit_stops_as_ambiguous(self) -> None:
        decision = self.decide(duration=240.0, strong_orbit=False, shots=3)
        self.assertEqual((decision["status"], decision["code"]), ("STOP", "AUTO_PROFILE_AMBIGUOUS"))

    def test_capture_preflight_failure_stops_before_profile_selection(self) -> None:
        decision = self.decide(duration=30.0, strong_orbit=True, capture_pass=False)
        self.assertEqual(decision["code"], "AUTO_CAPTURE_PREFLIGHT_FAILED")
        self.assertIsNone(decision["profile"])

    def test_pose_gate_failure_stops_before_profile_selection(self) -> None:
        decision = self.decide(duration=300.0, strong_orbit=False, pose_pass=False)
        self.assertEqual(decision["code"], "AUTO_POSE_PILOT_FAILED")
        self.assertIsNone(decision["profile"])

    def test_mismatched_model_counts_fail_closed(self) -> None:
        decision = MODULE.decide_profile(
            probe(300.0),
            frame_quality(),
            reconstruction(),
            convergence(False, registered=174),
        )
        self.assertEqual(decision["code"], "AUTO_EVIDENCE_INVALID")

    def test_same_count_different_model_hash_fails_closed(self) -> None:
        wrong_model = convergence(False)
        wrong_model["images_text_sha256"] = "f" * 64
        decision = MODULE.decide_profile(
            probe(300.0),
            frame_quality(),
            reconstruction(),
            wrong_model,
        )
        self.assertEqual(decision["code"], "AUTO_EVIDENCE_INVALID")
        self.assertIn("does not match", decision["reason"])

    def test_missing_shot_evidence_fails_closed(self) -> None:
        report = reconstruction()
        report["shot_manifest"] = None
        decision = MODULE.decide_profile(
            probe(300.0),
            frame_quality(),
            report,
            convergence(False),
        )
        self.assertEqual(decision["code"], "AUTO_SHOT_EVIDENCE_REQUIRED")

    def test_cli_measures_model_text_and_writes_bound_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "probe": root / "video_probe.json",
                "frames": root / "frame_quality.json",
                "report": root / "reconstruction_report.json",
                "images": root / "images.txt",
                "points": root / "points3D.txt",
                "output": root / "auto_decision.json",
            }
            paths["probe"].write_text(json.dumps(probe(20.5)), encoding="utf-8")
            paths["frames"].write_text(json.dumps(frame_quality()), encoding="utf-8")
            image_lines: list[str] = ["# synthetic inward-looking cameras"]
            for index in range(1, 31):
                angle = 2.0 * math.pi * (index - 1) / 30.0
                image_lines.extend(
                    [
                        f"{index} {math.cos(angle / 2.0):.9f} 0 {math.sin(angle / 2.0):.9f} 0 0 0 5 1 frame_{index:06d}.jpg",
                        "",
                    ]
                )
            paths["images"].write_text("\n".join(image_lines) + "\n", encoding="utf-8")
            point_lines = [
                f"{index} {((index % 3) - 1) * 0.01:.4f} {((index % 5) - 2) * 0.01:.4f} 0 100 100 100 0.1 1 0"
                for index in range(1, 21)
            ]
            paths["points"].write_text("\n".join(point_lines) + "\n", encoding="utf-8")
            report = reconstruction(loop_closed=True, registered=30, points=20)
            report["images_text_sha256"] = hashlib.sha256(paths["images"].read_bytes()).hexdigest()
            report["points3d_text_sha256"] = hashlib.sha256(paths["points"].read_bytes()).hexdigest()
            paths["report"].write_text(json.dumps(report), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--video-probe",
                    str(paths["probe"]),
                    "--frame-quality",
                    str(paths["frames"]),
                    "--reconstruction-report",
                    str(paths["report"]),
                    "--images-text",
                    str(paths["images"]),
                    "--points3d-text",
                    str(paths["points"]),
                    "--json",
                    str(paths["output"]),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision = json.loads(paths["output"].read_text(encoding="utf-8"))
            self.assertEqual((decision["status"], decision["profile"]), ("PASS", "Object"))
            self.assertEqual(decision["evidence"]["view_convergence"]["registered_views"], 30)


if __name__ == "__main__":
    unittest.main()
