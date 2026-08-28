#!/usr/bin/env python3
"""Select sharp, well-exposed, motion-useful frames with temporal coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


SELECTOR_VERSION = "2.4-max-cumulative-flow"
SHOT_DETECTOR_VERSION = "1.1-temporal-nms"
HARD_CUT_MIN_LUMINANCE_DELTA = 0.08
HARD_CUT_MIN_FLOW = 0.04
HARD_CUT_MIN_HOMOGRAPHY_RESIDUAL = 0.01
HARD_CUT_NMS_RADIUS = 3
HARD_CUT_MIN_ISOLATION_RATIO = 2.0
SUBTLE_PULSE_MIN_LUMINANCE_DELTA = 0.05
SUBTLE_PULSE_MIN_FLOW = 0.015
SUBTLE_PULSE_MAX_HOMOGRAPHY_RESIDUAL = 0.005
SUBTLE_RESET_MAX_LUMINANCE_DELTA = 0.015
SUBTLE_RESET_MAX_FLOW = 0.001
SUBTLE_RESET_MAX_HOMOGRAPHY_RESIDUAL = 0.0003
SUBTLE_PULSE_MIN_ISOLATION_RATIO = 5.0


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L")
        gray.thumbnail((640, 360), Image.Resampling.BILINEAR)
        return np.asarray(gray, dtype=np.uint8)


def score_image(gray: np.ndarray) -> dict[str, float]:
    pixels = gray.astype(np.float32)
    center = pixels[1:-1, 1:-1]
    laplacian = (
        pixels[:-2, 1:-1]
        + pixels[2:, 1:-1]
        + pixels[1:-1, :-2]
        + pixels[1:-1, 2:]
        - 4.0 * center
    )
    sharpness = float(np.var(laplacian))
    brightness = float(np.mean(pixels))
    contrast = float(np.std(pixels))
    clipped_dark = float(np.mean(pixels <= 5.0))
    clipped_bright = float(np.mean(pixels >= 250.0))

    exposure_penalty = max(0.0, 35.0 - brightness) + max(0.0, brightness - 220.0)
    contrast_penalty = max(0.0, 18.0 - contrast)
    clipping_penalty = 5.0 * max(0.0, clipped_dark + clipped_bright - 0.35)
    quality = (
        math.log1p(sharpness)
        - 0.035 * exposure_penalty
        - 0.02 * contrast_penalty
        - clipping_penalty
    )
    return {
        "sharpness": round(sharpness, 4),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "clipped_dark_fraction": round(clipped_dark, 6),
        "clipped_bright_fraction": round(clipped_bright, 6),
        "quality": round(quality, 6),
    }


def measure_motion(previous: np.ndarray, current: np.ndarray) -> dict[str, float | bool | int]:
    if previous.shape != current.shape:
        current = cv2.resize(current, (previous.shape[1], previous.shape[0]))

    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=600,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if points is None or len(points) < 20:
        return {
            "tracked_features": 0,
            "track_fraction": 0.0,
            "median_flow_normalized": 0.0,
            "homography_residual_normalized": 0.0,
            "near_duplicate": False,
            "low_parallax": False,
            "excessive_motion": False,
            "tracking_failed": True,
        }

    tracked, status, _errors = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None:
        valid_previous = np.empty((0, 2), dtype=np.float32)
        valid_current = np.empty((0, 2), dtype=np.float32)
    else:
        valid = status.reshape(-1).astype(bool)
        valid_previous = points.reshape(-1, 2)[valid]
        valid_current = tracked.reshape(-1, 2)[valid]

    diagonal = max(1.0, math.hypot(previous.shape[1], previous.shape[0]))
    tracked_count = len(valid_previous)
    track_fraction = tracked_count / max(1, len(points))
    if tracked_count < 15:
        return {
            "tracked_features": tracked_count,
            "track_fraction": round(track_fraction, 6),
            "median_flow_normalized": 0.0,
            "homography_residual_normalized": 0.0,
            "near_duplicate": False,
            "low_parallax": False,
            "excessive_motion": False,
            "tracking_failed": True,
        }

    flow = np.linalg.norm(valid_current - valid_previous, axis=1)
    median_flow = float(np.median(flow) / diagonal)
    homography, _inliers = cv2.findHomography(
        valid_previous,
        valid_current,
        cv2.RANSAC,
        3.0,
    )
    if homography is None:
        return {
            "tracked_features": tracked_count,
            "track_fraction": round(track_fraction, 6),
            "median_flow_normalized": round(median_flow, 8),
            "homography_residual_normalized": 0.0,
            "near_duplicate": False,
            "low_parallax": False,
            "excessive_motion": False,
            "tracking_failed": True,
        }

    projected = cv2.perspectiveTransform(
        valid_previous.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - valid_current, axis=1)
    # The dominant plane may be a wall or background while a smaller vehicle or
    # foreground layer supplies the useful parallax. A robust upper-tail sample
    # preserves that evidence without letting one bad optical-flow track decide.
    residual = float(np.percentile(errors, 90) / diagonal)

    near_duplicate = median_flow < 0.001 and residual < 0.0005
    # A rotating pinhole camera is explained almost perfectly by one homography.
    # Translation through a real 3D scene leaves a measurable depth-dependent residual.
    low_parallax = median_flow > 0.003 and residual < 0.0003
    excessive_motion = median_flow > 0.18 or track_fraction < 0.25
    return {
        "tracked_features": tracked_count,
        "track_fraction": round(track_fraction, 6),
        "median_flow_normalized": round(median_flow, 8),
        "homography_residual_normalized": round(residual, 8),
        "near_duplicate": near_duplicate,
        "low_parallax": low_parallax,
        "excessive_motion": excessive_motion,
        "tracking_failed": False,
    }


def luminance_delta(previous: np.ndarray, current: np.ndarray) -> float:
    if previous.shape != current.shape:
        current = cv2.resize(current, (previous.shape[1], previous.shape[0]))
    difference = np.abs(previous.astype(np.float32) - current.astype(np.float32))
    return round(float(np.mean(difference) / 255.0), 8)


def validated_edge_metrics(
    edge: dict[str, object],
) -> tuple[dict[str, float], bool, bool]:
    numeric_bounds = {
        "luminance_delta_normalized": (0.0, 1.0),
        "median_flow_normalized": (0.0, None),
        "homography_residual_normalized": (0.0, None),
        "track_fraction": (0.0, 1.0),
    }
    metrics: dict[str, float] = {}
    for name, (minimum, maximum) in numeric_bounds.items():
        value = edge.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Hard-cut {name} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < minimum or (
            maximum is not None and number > maximum
        ):
            raise ValueError(f"Hard-cut {name} is outside its valid range")
        metrics[name] = number
    for name in ("tracking_failed", "excessive_motion"):
        if not isinstance(edge.get(name), bool):
            raise ValueError(f"Hard-cut {name} must be boolean")
    return metrics, edge["tracking_failed"], edge["excessive_motion"]


def is_hard_cut(edge: dict[str, object]) -> bool:
    metrics, tracking_failed, excessive_motion = validated_edge_metrics(edge)

    luminance_discontinuity = (
        metrics["luminance_delta_normalized"] >= HARD_CUT_MIN_LUMINANCE_DELTA
    )
    motion_discontinuity = (
        tracking_failed
        or excessive_motion
        or (
            metrics["median_flow_normalized"] >= HARD_CUT_MIN_FLOW
            and metrics["homography_residual_normalized"]
            >= HARD_CUT_MIN_HOMOGRAPHY_RESIDUAL
        )
    )
    return luminance_discontinuity and motion_discontinuity


def discontinuity_score(edge: dict[str, object]) -> float:
    metrics, tracking_failed, excessive_motion = validated_edge_metrics(edge)
    score = max(
        metrics["median_flow_normalized"] / HARD_CUT_MIN_FLOW,
        metrics["homography_residual_normalized"] / HARD_CUT_MIN_HOMOGRAPHY_RESIDUAL,
    )
    if tracking_failed or excessive_motion:
        score = max(score, 2.0)
    return score


def detected_cut_indices(adjacent_motion: list[dict[str, object]]) -> list[int]:
    if not adjacent_motion:
        return []
    scores = [discontinuity_score(edge) for edge in adjacent_motion]
    cuts: set[int] = set()
    for index in range(1, len(adjacent_motion)):
        if not is_hard_cut(adjacent_motion[index]):
            continue
        neighbors = scores[
            max(1, index - HARD_CUT_NMS_RADIUS) : index
        ] + scores[index + 1 : index + HARD_CUT_NMS_RADIUS + 1]
        neighbor_peak = max(neighbors, default=0.0)
        if scores[index] >= HARD_CUT_MIN_ISOLATION_RATIO * neighbor_peak:
            cuts.add(index)

    for boundary in range(2, len(adjacent_motion) - 2):
        pulse = adjacent_motion[boundary - 1]
        reset = adjacent_motion[boundary]
        pulse_metrics, pulse_failed, pulse_excessive = validated_edge_metrics(pulse)
        reset_metrics, reset_failed, reset_excessive = validated_edge_metrics(reset)
        if pulse_failed or pulse_excessive or reset_failed or reset_excessive:
            continue
        if not (
            pulse_metrics["luminance_delta_normalized"]
            >= SUBTLE_PULSE_MIN_LUMINANCE_DELTA
            and pulse_metrics["median_flow_normalized"] >= SUBTLE_PULSE_MIN_FLOW
            and pulse_metrics["homography_residual_normalized"]
            <= SUBTLE_PULSE_MAX_HOMOGRAPHY_RESIDUAL
            and reset_metrics["luminance_delta_normalized"]
            <= SUBTLE_RESET_MAX_LUMINANCE_DELTA
            and reset_metrics["median_flow_normalized"] <= SUBTLE_RESET_MAX_FLOW
            and reset_metrics["homography_residual_normalized"]
            <= SUBTLE_RESET_MAX_HOMOGRAPHY_RESIDUAL
        ):
            continue
        neighbor_flows = [
            validated_edge_metrics(adjacent_motion[index])[0]["median_flow_normalized"]
            for index in range(max(1, boundary - 4), min(len(adjacent_motion), boundary + 3))
            if index not in (boundary - 1, boundary)
        ]
        if pulse_metrics["median_flow_normalized"] >= (
            SUBTLE_PULSE_MIN_ISOLATION_RATIO * max(neighbor_flows, default=0.0)
        ):
            cuts.add(boundary)
    return sorted(cuts)


def frame_names_sha256(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def validate_shot_manifest(
    manifest: dict[str, object], candidate_names: list[str], selected_names: list[str]
) -> None:
    if (
        not candidate_names
        or any(not isinstance(name, str) or not name for name in candidate_names)
        or len(candidate_names) != len(set(candidate_names))
    ):
        raise ValueError("Candidate frame names must be non-empty and unique")
    if (
        any(not isinstance(name, str) or not name for name in selected_names)
        or len(selected_names) != len(set(selected_names))
    ):
        raise ValueError("Selected frame names must be unique")
    selected_set = set(selected_names)
    ordered_selected = [name for name in candidate_names if name in selected_set]
    if ordered_selected != selected_names:
        raise ValueError("Selected frame names must be a candidate-ordered subset")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "stitched_video_shots":
        raise ValueError("Shot manifest schema or kind is invalid")
    if manifest.get("detector_version") != SHOT_DETECTOR_VERSION:
        raise ValueError("Shot manifest detector version is invalid")
    if manifest.get("candidate_frame_count") != len(candidate_names):
        raise ValueError("Shot manifest candidate count does not match the candidate names")
    if manifest.get("selected_frame_count") != len(selected_names):
        raise ValueError("Shot manifest selected count does not match the selected names")
    if manifest.get("candidate_frame_names_sha256") != frame_names_sha256(candidate_names):
        raise ValueError("Shot manifest candidate frame-name hash does not match")
    if manifest.get("selected_frame_names_sha256") != frame_names_sha256(selected_names):
        raise ValueError("Shot manifest selected frame-name hash does not match")

    shots = manifest.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("Shot manifest shots must be a non-empty array")
    shot_ids = manifest.get("shot_ids")
    if shot_ids != [shot.get("id") for shot in shots if isinstance(shot, dict)]:
        raise ValueError("Shot manifest shot_ids do not match its shot ranges")
    if (
        not isinstance(shot_ids, list)
        or any(not isinstance(shot_id, str) or not shot_id for shot_id in shot_ids)
        or len(shot_ids) != len(set(shot_ids))
    ):
        raise ValueError("Shot manifest shot ids must be non-empty and unique")
    expected_first = 1
    assigned_selected: list[str] = []
    for shot in shots:
        if not isinstance(shot, dict):
            raise ValueError("Each shot range must be an object")
        first = shot.get("first_candidate_index")
        last = shot.get("last_candidate_index")
        if (
            not isinstance(first, int)
            or isinstance(first, bool)
            or not isinstance(last, int)
            or isinstance(last, bool)
            or first != expected_first
            or last < first
            or last > len(candidate_names)
        ):
            raise ValueError("Shot ranges must be ordered, contiguous, and in bounds")
        if shot.get("first_candidate_name") != candidate_names[first - 1]:
            raise ValueError("Shot first candidate name does not match its range")
        if shot.get("last_candidate_name") != candidate_names[last - 1]:
            raise ValueError("Shot last candidate name does not match its range")
        expected_selected = [
            name for name in candidate_names[first - 1 : last] if name in selected_set
        ]
        if shot.get("selected_frame_names") != expected_selected:
            raise ValueError("Shot selected frame names do not match its range")
        assigned_selected.extend(expected_selected)
        expected_first = last + 1
    if expected_first != len(candidate_names) + 1 or assigned_selected != selected_names:
        raise ValueError("Shot ranges do not cover the exact candidate and selected frame sets")

    hard_cuts = manifest.get("hard_cuts")
    if not isinstance(hard_cuts, list) or len(hard_cuts) != len(shots) - 1:
        raise ValueError("Hard-cut evidence must match every boundary between shot ranges")
    for cut, next_shot in zip(hard_cuts, shots[1:]):
        if not isinstance(cut, dict) or not isinstance(next_shot, dict):
            raise ValueError("Hard-cut evidence must contain objects")
        before = next_shot["first_candidate_index"]
        if (
            cut.get("before_candidate_index") != before
            or cut.get("previous_candidate_name") != candidate_names[before - 2]
            or cut.get("next_candidate_name") != candidate_names[before - 1]
        ):
            raise ValueError("Hard-cut evidence does not match its shot boundary")
        try:
            cut_kind = cut.get("detection_kind")
            metrics, tracking_failed, excessive_motion = validated_edge_metrics(cut)
            if cut_kind == "isolated_discontinuity":
                neighbor_peak = cut.get("neighbor_peak_discontinuity_score")
                if (
                    not isinstance(neighbor_peak, (int, float))
                    or isinstance(neighbor_peak, bool)
                    or not math.isfinite(float(neighbor_peak))
                    or float(neighbor_peak) < 0.0
                    or not is_hard_cut(cut)
                    or discontinuity_score(cut)
                    < HARD_CUT_MIN_ISOLATION_RATIO * float(neighbor_peak)
                ):
                    raise ValueError("Isolated hard-cut metrics do not prove a cut")
            elif cut_kind == "pulse_to_reset":
                preceding = cut.get("preceding_edge")
                neighbor_peak = cut.get("neighbor_peak_flow_normalized")
                if not isinstance(preceding, dict):
                    raise ValueError("Pulse-to-reset evidence lacks its preceding edge")
                pulse_metrics, pulse_failed, pulse_excessive = validated_edge_metrics(preceding)
                if (
                    preceding.get("before_candidate_index") != before - 1
                    or preceding.get("previous_candidate_name")
                    != candidate_names[before - 3]
                    or preceding.get("next_candidate_name") != candidate_names[before - 2]
                    or not isinstance(neighbor_peak, (int, float))
                    or isinstance(neighbor_peak, bool)
                    or not math.isfinite(float(neighbor_peak))
                    or float(neighbor_peak) < 0.0
                    or pulse_failed
                    or pulse_excessive
                    or tracking_failed
                    or excessive_motion
                    or pulse_metrics["luminance_delta_normalized"]
                    < SUBTLE_PULSE_MIN_LUMINANCE_DELTA
                    or pulse_metrics["median_flow_normalized"] < SUBTLE_PULSE_MIN_FLOW
                    or pulse_metrics["homography_residual_normalized"]
                    > SUBTLE_PULSE_MAX_HOMOGRAPHY_RESIDUAL
                    or metrics["luminance_delta_normalized"]
                    > SUBTLE_RESET_MAX_LUMINANCE_DELTA
                    or metrics["median_flow_normalized"] > SUBTLE_RESET_MAX_FLOW
                    or metrics["homography_residual_normalized"]
                    > SUBTLE_RESET_MAX_HOMOGRAPHY_RESIDUAL
                    or pulse_metrics["median_flow_normalized"]
                    < SUBTLE_PULSE_MIN_ISOLATION_RATIO * float(neighbor_peak)
                ):
                    raise ValueError("Pulse-to-reset metrics do not prove a cut")
            else:
                raise ValueError("Hard-cut detection kind is invalid")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Hard-cut metrics are malformed") from error


def build_shot_manifest(
    candidate_names: list[str],
    selected_names: list[str],
    adjacent_motion: list[dict[str, object]],
) -> dict[str, object]:
    if len(adjacent_motion) != len(candidate_names):
        raise ValueError("Adjacent motion evidence must match the candidate frame count")
    cut_indices = detected_cut_indices(adjacent_motion)
    boundaries = [0, *cut_indices, len(candidate_names)]
    selected_set = set(selected_names)
    shots: list[dict[str, object]] = []
    for shot_number, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        shots.append(
            {
                "id": f"shot_{shot_number:02d}",
                "first_candidate_index": start + 1,
                "last_candidate_index": end,
                "first_candidate_name": candidate_names[start],
                "last_candidate_name": candidate_names[end - 1],
                "selected_frame_names": [
                    name for name in candidate_names[start:end] if name in selected_set
                ],
            }
        )
    hard_cuts = []
    scores = [discontinuity_score(edge) for edge in adjacent_motion]
    for index in cut_indices:
        edge = adjacent_motion[index]
        evidence: dict[str, object] = {
            "before_candidate_index": index + 1,
            "previous_candidate_name": candidate_names[index - 1],
            "next_candidate_name": candidate_names[index],
            "luminance_delta_normalized": edge["luminance_delta_normalized"],
            "median_flow_normalized": edge["median_flow_normalized"],
            "homography_residual_normalized": edge["homography_residual_normalized"],
            "track_fraction": edge["track_fraction"],
            "tracking_failed": edge["tracking_failed"],
            "excessive_motion": edge["excessive_motion"],
        }
        neighbors = scores[
            max(1, index - HARD_CUT_NMS_RADIUS) : index
        ] + scores[index + 1 : index + HARD_CUT_NMS_RADIUS + 1]
        neighbor_peak_score = max(neighbors, default=0.0)
        if is_hard_cut(edge) and scores[index] >= (
            HARD_CUT_MIN_ISOLATION_RATIO * neighbor_peak_score
        ):
            evidence["detection_kind"] = "isolated_discontinuity"
            evidence["neighbor_peak_discontinuity_score"] = round(neighbor_peak_score, 8)
        else:
            pulse = adjacent_motion[index - 1]
            neighbor_flows = [
                validated_edge_metrics(adjacent_motion[neighbor])[0][
                    "median_flow_normalized"
                ]
                for neighbor in range(
                    max(1, index - 4), min(len(adjacent_motion), index + 3)
                )
                if neighbor not in (index - 1, index)
            ]
            evidence["detection_kind"] = "pulse_to_reset"
            evidence["neighbor_peak_flow_normalized"] = round(
                max(neighbor_flows, default=0.0), 8
            )
            evidence["preceding_edge"] = {
                "before_candidate_index": index,
                "previous_candidate_name": candidate_names[index - 2],
                "next_candidate_name": candidate_names[index - 1],
                "luminance_delta_normalized": pulse["luminance_delta_normalized"],
                "median_flow_normalized": pulse["median_flow_normalized"],
                "homography_residual_normalized": pulse["homography_residual_normalized"],
                "track_fraction": pulse["track_fraction"],
                "tracking_failed": pulse["tracking_failed"],
                "excessive_motion": pulse["excessive_motion"],
            }
        hard_cuts.append(evidence)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "kind": "stitched_video_shots",
        "detector_version": SHOT_DETECTOR_VERSION,
        "candidate_frame_count": len(candidate_names),
        "candidate_frame_names_sha256": frame_names_sha256(candidate_names),
        "selected_frame_count": len(selected_names),
        "selected_frame_names_sha256": frame_names_sha256(selected_names),
        "shot_ids": [shot["id"] for shot in shots],
        "hard_cuts": hard_cuts,
        "shots": shots,
    }
    validate_shot_manifest(manifest, candidate_names, selected_names)
    return manifest


def build_sampling_bins(
    frame_count: int,
    target: int,
    adjacent_motion: list[dict[str, object]],
    max_cumulative_flow: float,
) -> list[tuple[int, int]]:
    """Keep the legacy temporal bins and subdivide only sustained high-flow spans."""
    temporal_boundaries = {
        math.floor(bin_index * frame_count / target)
        for bin_index in range(target + 1)
    }
    if max_cumulative_flow <= 0.0:
        ordered = sorted(temporal_boundaries)
        return list(zip(ordered[:-1], ordered[1:]))

    boundaries = set(temporal_boundaries)
    cumulative_flow = 0.0
    for index in range(1, frame_count):
        if index in temporal_boundaries:
            cumulative_flow = 0.0

        edge = adjacent_motion[index]
        if bool(edge["tracking_failed"]) or bool(edge["excessive_motion"]):
            boundaries.add(index)
            cumulative_flow = 0.0
            continue

        flow = max(0.0, float(edge["median_flow_normalized"]))
        cumulative_flow += min(flow, max_cumulative_flow)
        if cumulative_flow >= max_cumulative_flow:
            boundaries.add(index)
            cumulative_flow = 0.0

    ordered = sorted(boundaries)
    return [
        (start, end)
        for start, end in zip(ordered[:-1], ordered[1:])
        if end > start
    ]


def build_contact_sheet(paths: list[Path], destination: Path) -> None:
    sample_count = min(36, len(paths))
    if not sample_count:
        return

    indices = np.linspace(0, len(paths) - 1, sample_count, dtype=int)
    tile_width, tile_height = 320, 180
    columns = 6
    rows = math.ceil(sample_count / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "black")
    draw = ImageDraw.Draw(sheet)

    for slot, index in enumerate(indices):
        path = paths[int(index)]
        with Image.open(path) as source:
            frame = source.convert("RGB")
            frame.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x = (slot % columns) * tile_width
        y = (slot // columns) * tile_height
        sheet.paste(frame, (x, y))
        draw.rectangle((x, y + tile_height - 20, x + 152, y + tile_height), fill="black")
        draw.text((x + 5, y + tile_height - 17), path.stem, fill="white")

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument(
        "--profile",
        choices=("Object", "Walkthrough", "House", "AerialExterior"),
        required=True,
    )
    parser.add_argument("--blur-percentile", type=float, default=20.0)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--require-motion", action="store_true")
    parser.add_argument("--max-cumulative-flow", type=float, default=0.0)
    args = parser.parse_args()

    frames = sorted(args.input.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"No JPEG frames found in {args.input}")
    if args.target < 2:
        raise SystemExit("--target must be at least 2")
    if args.max_cumulative_flow < 0.0:
        raise SystemExit("--max-cumulative-flow cannot be negative")

    target = min(args.target, len(frames))
    scored: list[dict[str, object]] = []
    grays: list[np.ndarray] = []
    adjacent_motion: list[dict[str, object]] = [
        {
            "tracked_features": 0,
            "track_fraction": 0.0,
            "median_flow_normalized": 0.0,
            "homography_residual_normalized": 0.0,
            "tracking_failed": False,
            "excessive_motion": False,
            "luminance_delta_normalized": 0.0,
            "hard_cut": False,
        }
        for _ in frames
    ]
    for index, path in enumerate(frames):
        gray = load_gray(path)
        grays.append(gray)
        record: dict[str, object] = {
            "index": index,
            "filename": path.name,
            **score_image(gray),
            "tracked_features": 0,
            "track_fraction": 0.0,
            "median_flow_normalized": 0.0,
            "homography_residual_normalized": 0.0,
            "near_duplicate": False,
            "low_parallax": False,
            "excessive_motion": False,
            "tracking_failed": False,
            "adjacent_luminance_delta_normalized": 0.0,
            "adjacent_track_fraction": 0.0,
            "adjacent_median_flow_normalized": 0.0,
            "adjacent_homography_residual_normalized": 0.0,
            "adjacent_tracking_failed": False,
            "adjacent_excessive_motion": False,
            "adjacent_hard_cut": False,
        }
        scored.append(record)

    for index in range(1, len(frames)):
        delta = luminance_delta(grays[index - 1], grays[index])
        adjacent = measure_motion(grays[index - 1], grays[index])
        adjacent["luminance_delta_normalized"] = delta
        adjacent_motion[index] = adjacent
        scored[index]["adjacent_luminance_delta_normalized"] = delta
        scored[index]["adjacent_track_fraction"] = adjacent_motion[index]["track_fraction"]
        scored[index]["adjacent_median_flow_normalized"] = adjacent_motion[index][
            "median_flow_normalized"
        ]
        scored[index]["adjacent_homography_residual_normalized"] = adjacent_motion[index][
            "homography_residual_normalized"
        ]
        scored[index]["adjacent_tracking_failed"] = adjacent_motion[index]["tracking_failed"]
        scored[index]["adjacent_excessive_motion"] = adjacent_motion[index]["excessive_motion"]
    detected_cuts = set(detected_cut_indices(adjacent_motion))
    for index in range(1, len(frames)):
        adjacent_motion[index]["hard_cut"] = index in detected_cuts
        scored[index]["adjacent_hard_cut"] = index in detected_cuts

    blur_threshold = float(
        np.percentile([float(record["sharpness"]) for record in scored], args.blur_percentile)
    )
    motion_weight = 1.0 if args.profile == "Object" else 1.4
    for record in scored:
        record["below_blur_percentile"] = float(record["sharpness"]) < blur_threshold

    def update_selection_score(record: dict[str, object]) -> None:
        useful_flow = min(float(record["median_flow_normalized"]) / 0.02, 1.0)
        parallax_proxy = min(float(record["homography_residual_normalized"]) / 0.006, 1.0)
        motion_score = 0.7 * useful_flow + 0.3 * parallax_proxy
        penalty = 0.0
        if bool(record["near_duplicate"]):
            penalty += 2.0
        if bool(record["low_parallax"]):
            penalty += 1.5
        if bool(record["excessive_motion"]):
            penalty += 1.5
        if bool(record["tracking_failed"]):
            penalty += 1.0
        record["motion_score"] = round(motion_score, 6)
        record["selection_score"] = round(
            float(record["quality"]) + motion_weight * motion_score - penalty,
            6,
        )

    selected_indices: set[int] = set()
    previous_selected_index: int | None = None
    sampling_bins = build_sampling_bins(
        len(frames),
        target,
        adjacent_motion,
        args.max_cumulative_flow,
    )
    for start, end in sampling_bins:
        candidates = scored[start : max(start + 1, end)]
        for item in candidates:
            if previous_selected_index is not None:
                item.update(
                    measure_motion(
                        grays[previous_selected_index],
                        grays[int(item["index"])],
                    )
                )
            update_selection_score(item)
        usable = [
            item
            for item in candidates
            if not item["below_blur_percentile"]
            and not item["near_duplicate"]
            and not item["low_parallax"]
            and not item["excessive_motion"]
            and not item["tracking_failed"]
        ]
        sharp = [item for item in candidates if not item["below_blur_percentile"]]
        winner = max(usable or sharp or candidates, key=lambda item: item["selection_score"])
        previous_selected_index = int(winner["index"])
        selected_indices.add(previous_selected_index)

    selected = [record for record in scored if record["index"] in selected_indices]
    args.output.mkdir(parents=True, exist_ok=True)
    for old_file in args.output.glob("*.jpg"):
        old_file.unlink()
    for record in selected:
        shutil.copy2(args.input / str(record["filename"]), args.output / str(record["filename"]))
        record["selected"] = True
    for record in scored:
        record.setdefault("selected", False)

    candidate_names = [path.name for path in frames]
    selected_names = [str(record["filename"]) for record in selected]
    shot_manifest = build_shot_manifest(candidate_names, selected_names, adjacent_motion)

    selected_motion_failures = sum(
        1
        for record in selected
        if record["near_duplicate"]
        or record["low_parallax"]
        or record["excessive_motion"]
        or record["tracking_failed"]
    )
    motion_pass = selected_motion_failures <= max(2, math.ceil(len(selected) * 0.10))
    selected_clipping_failures = sum(
        1
        for record in selected
        if float(record["clipped_dark_fraction"]) + float(record["clipped_bright_fraction"]) > 0.65
    )
    exposure_pass = selected_clipping_failures <= max(2, math.ceil(len(selected) * 0.20))
    preflight_pass = motion_pass and exposure_pass

    report = {
        "selector_version": SELECTOR_VERSION,
        "profile": args.profile,
        "source_frames": len(frames),
        "requested_frames": args.target,
        "base_requested_frames": target,
        "selected_frames": len(selected),
        "adaptive_extra_frames": max(0, len(selected) - target),
        "max_cumulative_flow": args.max_cumulative_flow,
        "sampling_bins": len(sampling_bins),
        "blur_percentile": args.blur_percentile,
        "blur_threshold": round(blur_threshold, 4),
        "globally_blurry_frames": sum(
            1 for record in scored if record["below_blur_percentile"]
        ),
        "near_duplicate_frames": sum(1 for record in scored if record["near_duplicate"]),
        "low_parallax_frames": sum(1 for record in scored if record["low_parallax"]),
        "excessive_motion_frames": sum(1 for record in scored if record["excessive_motion"]),
        "tracking_failed_frames": sum(1 for record in scored if record["tracking_failed"]),
        "selected_motion_failures": selected_motion_failures,
        "selected_clipping_failures": selected_clipping_failures,
        "shot_manifest": shot_manifest,
        "preflight": {
            "motion_pass": motion_pass,
            "exposure_pass": exposure_pass,
            "overall_pass": preflight_pass,
        },
        "records": scored,
    }
    args.records.parent.mkdir(parents=True, exist_ok=True)
    args.records.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.records.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scored[0].keys())
        writer.writeheader()
        writer.writerows(scored)

    selected_paths = [args.output / str(record["filename"]) for record in selected]
    build_contact_sheet(selected_paths, args.contact_sheet)
    print(
        f"Selected {len(selected)} of {len(frames)} frames for {args.profile}; "
        f"motion preflight {'passed' if motion_pass else 'failed'}, "
        f"exposure preflight {'passed' if exposure_pass else 'failed'}"
    )
    if args.require_motion and not preflight_pass:
        raise SystemExit(
            "Capture preflight failed. Review frame_quality.json before reconstruction."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
