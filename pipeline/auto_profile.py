#!/usr/bin/env python3
"""Choose a SplatItUp888 capture profile from a bounded pose pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CLASSIFIER_VERSION = "1.0-fail-closed-pose-pilot"
ORBIT_MEDIAN_COSINE = 0.90
ORBIT_P10_COSINE = 0.35
ORBIT_POSITIVE_FRACTION = 0.95
ORBIT_MINIMUM_COVERAGE_DEGREES = 240.0
ORBIT_MINIMUM_SWEEP_DEGREES = 240.0
ORBIT_SCENE_EXTENT_PERCENTILE = 80.0
ORBIT_MINIMUM_RADIUS_RATIO = 1.05
ORBIT_MINIMUM_POINTS_INSIDE_FRACTION = 0.80
LONG_ORBIT_SECONDS = 120.0


class EvidenceError(ValueError):
    """Raised when pilot artifacts do not form one self-consistent contract."""


def stop(code: str, reason: str, evidence: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "classifier_version": CLASSIFIER_VERSION,
        "status": "STOP",
        "code": code,
        "profile": None,
        "reason": reason,
        "evidence": evidence or {},
    }


def selected(profile: str, reason: str, evidence: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "classifier_version": CLASSIFIER_VERSION,
        "status": "PASS",
        "code": "AUTO_PROFILE_SELECTED",
        "profile": profile,
        "reason": reason,
        "evidence": evidence,
    }


def require_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be a JSON object")
    return value


def require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise EvidenceError(f"{label} must be an integer")
    return value


def require_finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be finite")
    return result


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise EvidenceError(f"{label} must be a SHA256 string")
    try:
        int(value, 16)
    except ValueError as error:
        raise EvidenceError(f"{label} must be a SHA256 string") from error
    return value.lower()


def shot_count(report: dict[str, object]) -> int | None:
    manifest = report.get("shot_manifest")
    if manifest is None:
        return None
    manifest = require_dict(manifest, "reconstruction_report.shot_manifest")
    shot_ids = manifest.get("shot_ids")
    if not isinstance(shot_ids, list) or not shot_ids:
        raise EvidenceError("reconstruction_report.shot_manifest.shot_ids must be a non-empty array")
    if any(not isinstance(item, str) or not item for item in shot_ids):
        raise EvidenceError("shot_ids must contain non-empty strings")
    if len(shot_ids) != len(set(shot_ids)):
        raise EvidenceError("shot_ids must be unique")
    return len(shot_ids)


def decide_profile(
    probe: dict[str, object],
    frame_quality: dict[str, object],
    reconstruction: dict[str, object],
    convergence: dict[str, object],
) -> dict[str, object]:
    try:
        duration = require_finite(
            require_dict(probe.get("format"), "video_probe.format").get("duration"),
            "video_probe.format.duration",
        )
        if duration <= 0.0:
            raise EvidenceError("video duration must be positive")

        preflight = require_dict(frame_quality.get("preflight"), "frame_quality.preflight")
        selected_frames = require_int(frame_quality.get("selected_frames"), "frame_quality.selected_frames")
        gates = require_dict(reconstruction.get("quality_gates"), "reconstruction_report.quality_gates")
        selected_images = require_int(reconstruction.get("selected_images"), "reconstruction_report.selected_images")
        registered_images = require_int(reconstruction.get("registered_images"), "reconstruction_report.registered_images")
        sparse_points = require_int(reconstruction.get("points"), "reconstruction_report.points")
        if selected_frames != selected_images:
            raise EvidenceError("frame selection and pose pilot image counts do not match")
        if require_int(convergence.get("registered_views"), "convergence.registered_views") != registered_images:
            raise EvidenceError("images.txt does not match the pose pilot registered-image count")
        if require_int(convergence.get("sparse_points"), "convergence.sparse_points") != sparse_points:
            raise EvidenceError("points3D.txt does not match the pose pilot point count")
        expected_images_sha = require_sha256(
            reconstruction.get("images_text_sha256"), "reconstruction_report.images_text_sha256"
        )
        expected_points_sha = require_sha256(
            reconstruction.get("points3d_text_sha256"), "reconstruction_report.points3d_text_sha256"
        )
        actual_images_sha = require_sha256(
            convergence.get("images_text_sha256"), "convergence.images_text_sha256"
        )
        actual_points_sha = require_sha256(
            convergence.get("points3d_text_sha256"), "convergence.points3d_text_sha256"
        )
        if actual_images_sha != expected_images_sha or actual_points_sha != expected_points_sha:
            raise EvidenceError("the supplied COLMAP text model does not match the reconstruction report hashes")

        pilot_profile = reconstruction.get("profile")
        selection_profile = frame_quality.get("profile")
        if not isinstance(pilot_profile, str) or not pilot_profile:
            raise EvidenceError("reconstruction_report.profile is required")
        if selection_profile != pilot_profile:
            raise EvidenceError("frame selection and pose pilot profiles do not match")

        graph = require_dict(gates.get("verified_match_graph"), "quality_gates.verified_match_graph")
        continuity = require_dict(gates.get("trajectory_continuity"), "quality_gates.trajectory_continuity")
        closure = require_dict(gates.get("capture_loop_closure"), "quality_gates.capture_loop_closure")
        count_shots = shot_count(reconstruction)
        median_cosine = require_finite(convergence.get("median_cosine"), "convergence.median_cosine")
        p10_cosine = require_finite(convergence.get("p10_cosine"), "convergence.p10_cosine")
        positive_fraction = require_finite(
            convergence.get("positive_fraction"), "convergence.positive_fraction"
        )
        angular_coverage = require_finite(
            convergence.get("angular_coverage_degrees"), "convergence.angular_coverage_degrees"
        )
        angular_sweep = require_finite(
            convergence.get("angular_sweep_degrees"), "convergence.angular_sweep_degrees"
        )
        camera_median_radius = require_finite(
            convergence.get("camera_median_radius"), "convergence.camera_median_radius"
        )
        scene_extent_radius = require_finite(
            convergence.get("scene_point_p80_radius"), "convergence.scene_point_p80_radius"
        )
        radius_ratio = require_finite(
            convergence.get("camera_to_scene_extent_ratio"),
            "convergence.camera_to_scene_extent_ratio",
        )
        points_inside_fraction = require_finite(
            convergence.get("points_inside_camera_shell_fraction"),
            "convergence.points_inside_camera_shell_fraction",
        )
        if not (-1.0 <= median_cosine <= 1.0 and -1.0 <= p10_cosine <= 1.0):
            raise EvidenceError("view-convergence cosine is outside [-1, 1]")
        if not 0.0 <= positive_fraction <= 1.0:
            raise EvidenceError("view-convergence positive fraction is outside [0, 1]")
        if not 0.0 <= angular_coverage <= 360.0 or angular_sweep < 0.0:
            raise EvidenceError("camera-center angular coverage or sweep is invalid")
        if camera_median_radius <= 0.0 or scene_extent_radius <= 0.0 or radius_ratio <= 0.0:
            raise EvidenceError("camera-shell or sparse-scene extent is invalid")
        if not 0.0 <= points_inside_fraction <= 1.0:
            raise EvidenceError("points-inside-camera-shell fraction is outside [0, 1]")

        orbit_motion = (
            median_cosine >= ORBIT_MEDIAN_COSINE
            and p10_cosine >= ORBIT_P10_COSINE
            and positive_fraction >= ORBIT_POSITIVE_FRACTION
            and angular_coverage >= ORBIT_MINIMUM_COVERAGE_DEGREES
            and angular_sweep >= ORBIT_MINIMUM_SWEEP_DEGREES
        )
        outside_compact_subject = (
            radius_ratio >= ORBIT_MINIMUM_RADIUS_RATIO
            and points_inside_fraction >= ORBIT_MINIMUM_POINTS_INSIDE_FRACTION
        )
        strong_orbit = orbit_motion and outside_compact_subject
        loop_closed = closure.get("pass") is True
        evidence = {
            "pilot_profile": pilot_profile,
            "duration_seconds": round(duration, 3),
            "shot_count": count_shots,
            "selected_images": selected_images,
            "registered_images": registered_images,
            "registration_percent": reconstruction.get("registration_percent"),
            "sparse_points": sparse_points,
            "model_text_sha256": {
                "images": actual_images_sha,
                "points3d": actual_points_sha,
            },
            "capture_preflight_pass": preflight.get("overall_pass") is True,
            "pose_gates_pass": gates.get("overall_pass") is True,
            "verified_match_graph_pass": graph.get("pass") is True,
            "trajectory_continuity_pass": continuity.get("pass") is True,
            "loop_closure_pass": loop_closed,
            "loop_closure_applicable": closure.get("applicable"),
            "view_convergence": {
                "registered_views": registered_images,
                "median_cosine": round(median_cosine, 6),
                "p10_cosine": round(p10_cosine, 6),
                "positive_fraction": round(positive_fraction, 6),
                "angular_coverage_degrees": round(angular_coverage, 3),
                "angular_sweep_degrees": round(angular_sweep, 3),
                "camera_median_radius": round(camera_median_radius, 6),
                "scene_point_p80_radius": round(scene_extent_radius, 6),
                "camera_to_scene_extent_ratio": round(radius_ratio, 6),
                "points_inside_camera_shell_fraction": round(points_inside_fraction, 6),
                "inward_angular_motion": orbit_motion,
                "cameras_outside_compact_subject": outside_compact_subject,
                "strong_inward_orbit": strong_orbit,
            },
        }

        if preflight.get("overall_pass") is not True:
            return stop(
                "AUTO_CAPTURE_PREFLIGHT_FAILED",
                "Capture motion or exposure preflight failed; training is not authorized.",
                evidence,
            )
        if (
            gates.get("overall_pass") is not True
            or graph.get("pass") is not True
            or continuity.get("pass") is not True
        ):
            return stop(
                "AUTO_POSE_PILOT_FAILED",
                "The bounded camera solve is incomplete, disconnected, or discontinuous.",
                evidence,
            )
        if count_shots is None:
            return stop(
                "AUTO_SHOT_EVIDENCE_REQUIRED",
                "Shot-boundary evidence is absent, so single-shot and stitched capture profiles cannot be distinguished safely.",
                evidence,
            )

        if orbit_motion and not outside_compact_subject:
            return stop(
                "AUTO_PROFILE_AMBIGUOUS",
                "The cameras circle the sparse scene but are not proven outside a compact subject; an enclosing room loop must not be treated as an object orbit.",
                evidence,
            )

        if strong_orbit:
            if count_shots > 1 or duration > LONG_ORBIT_SECONDS:
                return selected(
                    "AerialExterior",
                    "The pilot is a long or multi-shot inward-looking orbit; use chronological global mapping.",
                    evidence,
                )
            if loop_closed:
                return selected(
                    "Object",
                    "The pilot is a short single-shot inward-looking closed orbit.",
                    evidence,
                )
            return stop(
                "AUTO_PROFILE_AMBIGUOUS",
                "A short inward-looking orbit lacks verified closure, so Object versus AerialExterior is unsafe to guess.",
                evidence,
            )

        if loop_closed:
            return selected(
                "House",
                "The pilot is a connected non-orbit route with verified loop closure.",
                evidence,
            )
        if count_shots == 1:
            return selected(
                "Walkthrough",
                "The pilot is a connected single-shot open route without orbit convergence.",
                evidence,
            )
        return stop(
            "AUTO_PROFILE_AMBIGUOUS",
            "A multi-shot non-orbit solve could be an exterior pass or a stitched walkthrough; choose the profile manually.",
            evidence,
        )
    except EvidenceError as error:
        return stop("AUTO_EVIDENCE_INVALID", str(error))


def quaternion_rotation(values: list[float]) -> np.ndarray:
    q = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise EvidenceError("images.txt contains an invalid camera quaternion")
    qw, qx, qy, qz = q / norm
    return np.asarray(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def read_registered_views(text: str) -> list[tuple[np.ndarray, np.ndarray]]:
    lines = text.splitlines()
    named_views: list[tuple[str, np.ndarray, np.ndarray]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=9)
        if len(parts) < 10:
            raise EvidenceError("images.txt contains a malformed camera record")
        rotation = quaternion_rotation([float(value) for value in parts[1:5]])
        translation = np.asarray([float(value) for value in parts[5:8]], dtype=np.float64)
        center = -(rotation.T @ translation)
        forward = rotation.T @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        named_views.append((parts[9], center, forward))
        if index < len(lines):
            index += 1
    if not named_views:
        raise EvidenceError("images.txt contains no registered cameras")
    named_views.sort(key=lambda item: item[0].casefold())
    return [(center, forward) for _name, center, forward in named_views]


def read_sparse_points(text: str) -> np.ndarray:
    points: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=4)
        if len(parts) < 4:
            raise EvidenceError("points3D.txt contains a malformed point record")
        points.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not points:
        raise EvidenceError("points3D.txt contains no sparse points")
    result = np.asarray(points, dtype=np.float64)
    if not np.isfinite(result).all():
        raise EvidenceError("points3D.txt contains non-finite coordinates")
    return result


def measure_view_convergence(images_text: Path, points_text: Path) -> dict[str, object]:
    images_bytes = images_text.read_bytes()
    points_bytes = points_text.read_bytes()
    views = read_registered_views(images_bytes.decode("utf-8-sig"))
    points = read_sparse_points(points_bytes.decode("utf-8-sig"))
    target = np.median(points, axis=0)
    cosines: list[float] = []
    offsets: list[np.ndarray] = []
    for center, forward in views:
        toward_target = target - center
        distance = float(np.linalg.norm(toward_target))
        if distance <= 1e-12:
            continue
        cosines.append(float(np.dot(forward, toward_target / distance)))
        offsets.append(center - target)
    if len(cosines) < max(10, math.ceil(len(views) * 0.9)):
        raise EvidenceError("too many camera centers coincide with the sparse-point centroid")
    values = np.asarray(cosines, dtype=np.float64)
    camera_offsets = np.asarray(offsets, dtype=np.float64)
    _left, _singular, axes = np.linalg.svd(camera_offsets, full_matrices=False)
    if axes.shape[0] < 2:
        raise EvidenceError("camera centers do not define an angular sweep plane")
    projected = np.column_stack((camera_offsets @ axes[0], camera_offsets @ axes[1]))
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    circular = np.sort((angles + 2.0 * math.pi) % (2.0 * math.pi))
    gaps = np.diff(np.concatenate((circular, circular[:1] + 2.0 * math.pi)))
    angular_coverage = 2.0 * math.pi - float(np.max(gaps))
    angular_steps = np.angle(np.exp(1j * np.diff(angles)))
    angular_sweep = float(np.sum(np.abs(angular_steps)))
    camera_radii = np.linalg.norm(camera_offsets, axis=1)
    point_radii = np.linalg.norm(points - target, axis=1)
    camera_median_radius = float(np.median(camera_radii))
    scene_extent_radius = float(np.percentile(point_radii, ORBIT_SCENE_EXTENT_PERCENTILE))
    if camera_median_radius <= 1e-12 or scene_extent_radius <= 1e-12:
        raise EvidenceError("camera-shell or sparse-scene extent is degenerate")
    return {
        "registered_views": len(views),
        "sparse_points": len(points),
        "images_text_sha256": hashlib.sha256(images_bytes).hexdigest(),
        "points3d_text_sha256": hashlib.sha256(points_bytes).hexdigest(),
        "median_cosine": float(np.median(values)),
        "p10_cosine": float(np.percentile(values, 10)),
        "positive_fraction": float(np.mean(values > 0.0)),
        "angular_coverage_degrees": math.degrees(angular_coverage),
        "angular_sweep_degrees": math.degrees(angular_sweep),
        "camera_median_radius": camera_median_radius,
        "scene_point_p80_radius": scene_extent_radius,
        "camera_to_scene_extent_ratio": camera_median_radius / scene_extent_radius,
        "points_inside_camera_shell_fraction": float(np.mean(point_radii <= camera_median_radius)),
    }


def read_json(path: Path) -> dict[str, object]:
    return require_dict(json.loads(path.read_text(encoding="utf-8-sig")), str(path))


def classify_files(
    video_probe: Path,
    frame_quality: Path,
    reconstruction_report: Path,
    images_text: Path,
    points_text: Path,
) -> dict[str, object]:
    convergence = measure_view_convergence(images_text, points_text)
    return decide_profile(
        read_json(video_probe),
        read_json(frame_quality),
        read_json(reconstruction_report),
        convergence,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-probe", type=Path, required=True)
    parser.add_argument("--frame-quality", type=Path, required=True)
    parser.add_argument("--reconstruction-report", type=Path, required=True)
    parser.add_argument("--images-text", type=Path, required=True)
    parser.add_argument("--points3d-text", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    try:
        decision = classify_files(
            args.video_probe,
            args.frame_quality,
            args.reconstruction_report,
            args.images_text,
            args.points3d_text,
        )
    except Exception as error:
        decision = stop("AUTO_EVIDENCE_INVALID", str(error))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.json.with_name(args.json.name + ".tmp")
    temporary.write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.json)
    print(f"{decision['code']}: {decision['reason']}")
    return 0 if decision["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
