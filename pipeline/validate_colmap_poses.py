#!/usr/bin/env python3
"""Validate COLMAP registration, match-graph continuity, closure, and trajectory."""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from pathlib import Path

import numpy as np


GATE_VERSION = "2.9-shot-edge-redundancy"
MINIMUM_SHOT_PAIR_EDGES = 3
MAX_IMAGE_ID = 2_147_483_647
WALKTHROUGH_MISSING_RUN_CAP = 24
FRAME_NAME_PATTERN = re.compile(r"^frame_(\d+)\.jpg$", re.IGNORECASE)


def quaternion_rotation(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def read_registered_images(path: Path) -> dict[int, dict[str, object]]:
    images: dict[int, dict[str, object]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=9)
        if len(parts) < 10:
            raise ValueError(f"Malformed COLMAP image line: {line}")
        image_id = int(parts[0])
        qvec = np.asarray([float(value) for value in parts[1:5]], dtype=np.float64)
        tvec = np.asarray([float(value) for value in parts[5:8]], dtype=np.float64)
        rotation = quaternion_rotation(qvec)
        center = -(rotation.T @ tvec)
        images[image_id] = {
            "image_id": image_id,
            "camera_id": int(parts[8]),
            "name": parts[9],
            "center": center,
        }
        if index < len(lines):
            index += 1  # The following line contains POINTS2D triples and may be blank.
    return images


def read_database(database: Path) -> tuple[dict[int, str], list[tuple[int, int, int]]]:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        names = {
            int(image_id): str(name)
            for image_id, name in connection.execute("SELECT image_id, name FROM images")
        }
        verified: list[tuple[int, int, int]] = []
        for pair_id, rows in connection.execute(
            "SELECT pair_id, rows FROM two_view_geometries WHERE rows >= 15"
        ):
            image_id2 = int(pair_id) % MAX_IMAGE_ID
            image_id1 = (int(pair_id) - image_id2) // MAX_IMAGE_ID
            verified.append((image_id1, image_id2, int(rows)))
        return names, verified
    finally:
        connection.close()


def largest_component(
    nodes: set[int], edges: list[tuple[int, int, int]]
) -> tuple[int, int]:
    adjacency = {node: set() for node in nodes}
    edge_count = 0
    for first, second, _rows in edges:
        if first in adjacency and second in adjacency:
            adjacency[first].add(second)
            adjacency[second].add(first)
            edge_count += 1
    largest = 0
    remaining = set(nodes)
    while remaining:
        start = remaining.pop()
        stack = [start]
        size = 1
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    size += 1
        largest = max(largest, size)
    return largest, edge_count


def maximum_missing_run(selected_names: list[str], registered_names: set[str]) -> int:
    maximum = 0
    current = 0
    for name in selected_names:
        if name in registered_names:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def load_shot_manifest(
    path: Path, selected_names: list[str]
) -> tuple[dict[str, object], list[str], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Shot manifest must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("Shot manifest schema_version must be 1")
    if payload.get("kind") != "stitched_video_shots":
        raise ValueError("Shot manifest kind must be stitched_video_shots")

    candidate_count = payload.get("candidate_frame_count")
    if not isinstance(candidate_count, int) or isinstance(candidate_count, bool) or candidate_count < 1:
        raise ValueError("Shot manifest candidate_frame_count must be a positive integer")
    shots = payload.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("Shot manifest shots must be a non-empty array")

    shot_order: list[str] = []
    shot_ranges: list[tuple[int, int, str]] = []
    seen_ids: set[str] = set()
    expected_first = 1
    for shot in shots:
        if not isinstance(shot, dict):
            raise ValueError("Each shot manifest entry must be an object")
        shot_id = shot.get("id")
        first = shot.get("first_candidate_index")
        last = shot.get("last_candidate_index")
        if not isinstance(shot_id, str) or not shot_id.strip():
            raise ValueError("Each shot must have a non-empty string id")
        if shot_id in seen_ids:
            raise ValueError(f"Duplicate shot id: {shot_id}")
        if (
            not isinstance(first, int)
            or isinstance(first, bool)
            or not isinstance(last, int)
            or isinstance(last, bool)
            or first > last
        ):
            raise ValueError(f"Shot {shot_id} has an invalid candidate range")
        if first != expected_first:
            raise ValueError(
                f"Shot {shot_id} must start at candidate {expected_first}; got {first}"
            )
        seen_ids.add(shot_id)
        shot_order.append(shot_id)
        shot_ranges.append((first, last, shot_id))
        expected_first = last + 1
    if expected_first - 1 != candidate_count:
        raise ValueError(
            "Shot ranges must end at candidate_frame_count "
            f"({candidate_count}); got {expected_first - 1}"
        )

    name_to_shot: dict[str, str] = {}
    range_index = 0
    for name in selected_names:
        match = FRAME_NAME_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f"Selected image does not use frame_N.jpg naming: {name}")
        candidate_index = int(match.group(1))
        while range_index < len(shot_ranges) and candidate_index > shot_ranges[range_index][1]:
            range_index += 1
        if (
            range_index >= len(shot_ranges)
            or candidate_index < shot_ranges[range_index][0]
        ):
            raise ValueError(f"Selected image is outside the declared shot ranges: {name}")
        name_to_shot[name] = shot_ranges[range_index][2]
    return payload, shot_order, name_to_shot


def maximum_missing_run_by_shot(
    selected_names: list[str],
    registered_names: set[str],
    name_to_shot: dict[str, str],
    shot_order: list[str],
) -> dict[str, int]:
    names_by_shot = {shot_id: [] for shot_id in shot_order}
    for name in selected_names:
        names_by_shot[name_to_shot[name]].append(name)
    return {
        shot_id: maximum_missing_run(names_by_shot[shot_id], registered_names)
        for shot_id in shot_order
    }


def maximum_missing_run_limit(profile: str, selected_count: int) -> int:
    if profile == "Object":
        return 5
    if profile == "House":
        return 10
    return min(WALKTHROUGH_MISSING_RUN_CAP, max(3, math.ceil(selected_count * 0.02)))


def trajectory_metrics(images: dict[int, dict[str, object]]) -> dict[str, object]:
    ordered = sorted(images.values(), key=lambda image: str(image["name"]).lower())
    centers = [np.asarray(image["center"], dtype=np.float64) for image in ordered]
    raw_steps = [float(np.linalg.norm(second - first)) for first, second in zip(centers, centers[1:])]
    raw_positive = [step for step in raw_steps if step > 1e-9]
    raw_median = float(np.median(raw_positive)) if raw_positive else 0.0
    isolated: list[int] = []
    if raw_median > 1e-9:
        index = 1
        while index < len(centers) - 1:
            island: list[int] = []
            for length in range(1, 4):
                end = index + length - 1
                if end + 1 >= len(centers):
                    break
                left = float(np.linalg.norm(centers[index] - centers[index - 1]))
                right = float(np.linalg.norm(centers[end + 1] - centers[end]))
                bridge = float(np.linalg.norm(centers[end + 1] - centers[index - 1]))
                if (
                    left > 20.0 * raw_median
                    and right > 20.0 * raw_median
                    and bridge <= 10.0 * raw_median
                ):
                    island = list(range(index, end + 1))
                    break
            if island:
                isolated.extend(island)
                index = island[-1] + 1
            else:
                index += 1

    isolated_set = set(isolated)
    filtered_centers = [center for index, center in enumerate(centers) if index not in isolated_set]
    steps = [
        float(np.linalg.norm(second - first))
        for first, second in zip(filtered_centers, filtered_centers[1:])
    ]
    positive = [step for step in steps if step > 1e-9]
    median = float(np.median(positive)) if positive else 0.0
    maximum = max(steps, default=0.0)
    ratio = maximum / median if median > 1e-9 else None
    raw_maximum = max(raw_steps, default=0.0)
    raw_ratio = raw_maximum / raw_median if raw_median > 1e-9 else None
    path_length = float(sum(steps))
    loop_distance = (
        float(np.linalg.norm(filtered_centers[-1] - filtered_centers[0]))
        if len(filtered_centers) >= 2
        else 0.0
    )
    return {
        "camera_count": len(centers),
        "validated_camera_count": len(filtered_centers),
        "excluded_isolated_poses": [str(ordered[index]["name"]) for index in isolated],
        "raw_maximum_to_median_step_ratio": round(raw_ratio, 4) if raw_ratio is not None else None,
        "median_step": round(median, 8),
        "maximum_step": round(maximum, 8),
        "maximum_to_median_step_ratio": round(ratio, 4) if ratio is not None else None,
        "path_length": round(path_length, 8),
        "first_to_last_distance": round(loop_distance, 8),
        "first_to_last_over_path": round(loop_distance / path_length, 6) if path_length > 1e-9 else None,
        "first_to_last_over_median_step": round(loop_distance / median, 4) if median > 1e-9 else None,
    }


def trajectory_metrics_by_shot(
    images: dict[int, dict[str, object]],
    name_to_shot: dict[str, str],
    shot_order: list[str],
) -> dict[str, object]:
    images_by_shot: dict[str, dict[int, dict[str, object]]] = {
        shot_id: {} for shot_id in shot_order
    }
    for image_id, image in images.items():
        shot_id = name_to_shot.get(str(image["name"]))
        if shot_id is not None:
            images_by_shot[shot_id][image_id] = image

    per_shot: dict[str, dict[str, object]] = {}
    excluded_names: list[str] = []
    raw_ratios: list[float] = []
    ratios: list[float] = []
    filtered_steps: list[float] = []
    path_length = 0.0
    validated_count = 0
    for shot_id in shot_order:
        shot_images = images_by_shot[shot_id]
        metrics = trajectory_metrics(shot_images)
        per_shot[shot_id] = metrics
        excluded = set(str(name) for name in metrics["excluded_isolated_poses"])
        excluded_names.extend(sorted(excluded))
        validated_count += int(metrics["validated_camera_count"])
        path_length += float(metrics["path_length"])
        if metrics["raw_maximum_to_median_step_ratio"] is not None:
            raw_ratios.append(float(metrics["raw_maximum_to_median_step_ratio"]))
        if metrics["maximum_to_median_step_ratio"] is not None:
            ratios.append(float(metrics["maximum_to_median_step_ratio"]))

        ordered = sorted(shot_images.values(), key=lambda image: str(image["name"]).lower())
        filtered_centers = [
            np.asarray(image["center"], dtype=np.float64)
            for image in ordered
            if str(image["name"]) not in excluded
        ]
        filtered_steps.extend(
            float(np.linalg.norm(second - first))
            for first, second in zip(filtered_centers, filtered_centers[1:])
        )

    positive_steps = [step for step in filtered_steps if step > 1e-9]
    median_step = float(np.median(positive_steps)) if positive_steps else 0.0
    maximum_step = max(filtered_steps, default=0.0)
    return {
        "camera_count": sum(len(group) for group in images_by_shot.values()),
        "validated_camera_count": validated_count,
        "excluded_isolated_poses": excluded_names,
        "raw_maximum_to_median_step_ratio": round(max(raw_ratios), 4) if raw_ratios else None,
        "median_step": round(median_step, 8),
        "maximum_step": round(maximum_step, 8),
        "maximum_to_median_step_ratio": round(max(ratios), 4) if ratios else None,
        "path_length": round(path_length, 8),
        "first_to_last_distance": None,
        "first_to_last_over_path": None,
        "first_to_last_over_median_step": None,
        "shot_aware": True,
        "ignored_cut_transitions": max(0, len(shot_order) - 1),
        "shots": per_shot,
    }


def shot_registration_metrics(
    selected_names: list[str],
    registered_names: set[str],
    name_to_shot: dict[str, str],
    shot_order: list[str],
    minimum_percent: float = 90.0,
) -> dict[str, object]:
    selected_by_shot = {shot_id: 0 for shot_id in shot_order}
    registered_by_shot = {shot_id: 0 for shot_id in shot_order}
    for name in selected_names:
        shot_id = name_to_shot[name]
        selected_by_shot[shot_id] += 1
        if name in registered_names:
            registered_by_shot[shot_id] += 1

    shots: dict[str, dict[str, object]] = {}
    for shot_id in shot_order:
        selected_count = selected_by_shot[shot_id]
        registered_count = registered_by_shot[shot_id]
        percent = 100.0 * registered_count / selected_count if selected_count else 0.0
        shots[shot_id] = {
            "pass": selected_count > 0 and percent >= minimum_percent,
            "selected_images": selected_count,
            "registered_images": registered_count,
            "actual_percent": round(percent, 3),
            "minimum_percent": minimum_percent,
        }
    return {
        "pass": all(bool(shot["pass"]) for shot in shots.values()),
        "minimum_percent": minimum_percent,
        "shots": shots,
    }


def shot_graph_metrics(
    model_images: dict[int, dict[str, object]],
    verified_edges: list[tuple[int, int, int]],
    name_to_shot: dict[str, str],
    shot_order: list[str],
) -> dict[str, object]:
    image_shots = {
        image_id: name_to_shot[str(image["name"])]
        for image_id, image in model_images.items()
        if str(image["name"]) in name_to_shot
    }
    connection_counts: dict[tuple[str, str], int] = {}
    for first, second, _rows in verified_edges:
        first_shot = image_shots.get(first)
        second_shot = image_shots.get(second)
        if first_shot is None or second_shot is None or first_shot == second_shot:
            continue
        pair = tuple(sorted((first_shot, second_shot)))
        connection_counts[pair] = connection_counts.get(pair, 0) + 1

    adjacency = {shot_id: set() for shot_id in shot_order}
    for (first_shot, second_shot), count in connection_counts.items():
        if count < MINIMUM_SHOT_PAIR_EDGES:
            continue
        adjacency[first_shot].add(second_shot)
        adjacency[second_shot].add(first_shot)

    largest = 0
    remaining = set(shot_order)
    while remaining:
        start = remaining.pop()
        stack = [start]
        size = 1
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    size += 1
        largest = max(largest, size)

    registered_shots = set(image_shots.values())
    connections = [
        {
            "first_shot": pair[0],
            "second_shot": pair[1],
            "verified_edges": count,
            "passes_minimum": count >= MINIMUM_SHOT_PAIR_EDGES,
        }
        for pair, count in sorted(connection_counts.items())
    ]
    return {
        "pass": (
            bool(shot_order)
            and largest == len(shot_order)
            and registered_shots == set(shot_order)
        ),
        "largest_component": largest,
        "shots": len(shot_order),
        "registered_shots": len(registered_shots),
        "verified_cross_shot_edges": sum(connection_counts.values()),
        "minimum_verified_edges_per_connection": MINIMUM_SHOT_PAIR_EDGES,
        "connections": connections,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-text", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("Object", "Walkthrough", "House", "AerialExterior"),
        required=True,
    )
    parser.add_argument("--registered", type=int, required=True)
    parser.add_argument("--points", type=int, required=True)
    parser.add_argument("--mean-track-length", type=float, required=True)
    parser.add_argument("--reprojection-error", type=float, required=True)
    parser.add_argument("--shot-manifest", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    model_images = read_registered_images(args.model_text / "images.txt")
    database_names, verified_edges = read_database(args.database)
    selected_names = sorted(path.name for path in args.selected_images.glob("*.jpg"))
    shot_manifest: dict[str, object] | None = None
    shot_order: list[str] = []
    name_to_shot: dict[str, str] = {}
    if args.shot_manifest is not None:
        shot_manifest, shot_order, name_to_shot = load_shot_manifest(
            args.shot_manifest, selected_names
        )
    registered_names = {str(image["name"]) for image in model_images.values()}
    selected_name_set = set(selected_names)
    database_name_set = set(database_names.values())
    model_database_id_mismatches = [
        {
            "image_id": image_id,
            "model_name": str(image["name"]),
            "database_name": database_names.get(image_id),
        }
        for image_id, image in sorted(model_images.items())
        if database_names.get(image_id) != str(image["name"])
    ]
    selected_count = len(selected_names)
    model_registered_count = len(model_images)
    registration_percent = 100.0 * model_registered_count / selected_count if selected_count else 0.0

    registered_ids = set(model_images)
    largest, verified_edge_count = largest_component(registered_ids, verified_edges)
    graph_connected = bool(registered_ids) and largest == len(registered_ids)
    missing_runs_by_shot = (
        maximum_missing_run_by_shot(
            selected_names, registered_names, name_to_shot, shot_order
        )
        if shot_manifest is not None
        else None
    )
    missing_run = (
        max(missing_runs_by_shot.values(), default=0)
        if missing_runs_by_shot is not None
        else maximum_missing_run(selected_names, registered_names)
    )

    name_to_id = {name: image_id for image_id, name in database_names.items()}
    closure_window = max(5, math.ceil(selected_count * 0.05))
    early_ids = {name_to_id[name] for name in selected_names[:closure_window] if name in name_to_id}
    late_ids = {name_to_id[name] for name in selected_names[-closure_window:] if name in name_to_id}
    multi_shot = shot_manifest is not None and len(shot_order) > 1
    closure_edges = [] if multi_shot else [
        {
            "first": database_names.get(first, str(first)),
            "second": database_names.get(second, str(second)),
            "inliers": rows,
        }
        for first, second, rows in verified_edges
        if first in registered_ids
        and second in registered_ids
        and ((first in early_ids and second in late_ids) or (second in early_ids and first in late_ids))
    ]

    trajectory = (
        trajectory_metrics_by_shot(model_images, name_to_shot, shot_order)
        if shot_manifest is not None
        else trajectory_metrics(model_images)
    )
    loop_path_ratio = trajectory["first_to_last_over_path"]
    loop_step_ratio = trajectory["first_to_last_over_median_step"]
    maximum_loop_steps = 30.0 if args.profile == "Object" else 10.0
    loop_closure_pass = (
        not multi_shot
        and bool(closure_edges)
        and loop_path_ratio is not None
        and float(loop_path_ratio) <= 0.15
        and loop_step_ratio is not None
        and float(loop_step_ratio) <= maximum_loop_steps
    )
    loop_closure_blocking = not multi_shot and args.profile in ("Object", "House")
    jump_limit = 20.0 if args.profile == "Object" else 25.0
    jump_ratio = trajectory["maximum_to_median_step_ratio"]
    excluded_poses = list(trajectory["excluded_isolated_poses"])
    isolated_limit = max(1, math.ceil(len(model_images) * 0.01))
    trajectory_pass = (
        jump_ratio is not None
        and float(jump_ratio) <= jump_limit
        and len(excluded_poses) <= isolated_limit
    )

    if args.profile == "Object":
        minimum_registration = 95.0
        minimum_points = 5_000
    else:
        minimum_registration = 90.0
        minimum_points = 20_000
    maximum_missing = maximum_missing_run_limit(args.profile, selected_count)
    missing_run_details: dict[str, dict[str, object]] | None = None
    missing_run_pass = missing_run <= maximum_missing
    if missing_runs_by_shot is not None:
        selected_counts_by_shot = {shot_id: 0 for shot_id in shot_order}
        for name in selected_names:
            selected_counts_by_shot[name_to_shot[name]] += 1
        missing_run_details = {}
        for shot_id in shot_order:
            shot_limit = maximum_missing_run_limit(
                args.profile, selected_counts_by_shot[shot_id]
            )
            actual = missing_runs_by_shot[shot_id]
            missing_run_details[shot_id] = {
                "pass": actual <= shot_limit,
                "actual": actual,
                "maximum": shot_limit,
            }
        missing_run_pass = all(
            bool(details["pass"]) for details in missing_run_details.values()
        )

    per_shot_registration = (
        shot_registration_metrics(
            selected_names, registered_names, name_to_shot, shot_order
        )
        if shot_manifest is not None
        else None
    )
    shot_graph = (
        shot_graph_metrics(model_images, verified_edges, name_to_shot, shot_order)
        if shot_manifest is not None
        else None
    )

    gates = {
        "model_integrity": {
            "pass": (
                args.registered == model_registered_count
                and registered_names <= selected_name_set
                and selected_name_set <= database_name_set
                and not model_database_id_mismatches
            ),
            "analyzer_registered_images": args.registered,
            "parsed_registered_images": model_registered_count,
            "registered_images_outside_selection": sorted(registered_names - selected_name_set),
            "selected_images_missing_from_database": sorted(selected_name_set - database_name_set),
            "model_database_id_mismatches": model_database_id_mismatches,
        },
        "registration": {
            "pass": registration_percent >= minimum_registration,
            "actual_percent": round(registration_percent, 3),
            "minimum_percent": minimum_registration,
        },
        "sparse_points": {
            "pass": args.points >= minimum_points,
            "actual": args.points,
            "minimum": minimum_points,
        },
        "mean_track_length": {
            "pass": args.mean_track_length >= 3.0,
            "actual": args.mean_track_length,
            "minimum": 3.0,
        },
        "reprojection_error": {
            "pass": args.reprojection_error <= 1.5,
            "actual_pixels": args.reprojection_error,
            "maximum_pixels": 1.5,
        },
        "verified_match_graph": {
            "pass": graph_connected,
            "largest_component": largest,
            "registered_images": len(registered_ids),
            "verified_edges": verified_edge_count,
        },
        "maximum_missing_run": {
            "pass": missing_run_pass,
            "actual": missing_run,
            "maximum": maximum_missing,
            "shot_aware": shot_manifest is not None,
            "shots": missing_run_details,
        },
        "capture_loop_closure": {
            "pass": loop_closure_pass,
            "blocking": loop_closure_blocking,
            "applicable": not multi_shot,
            "window_images": closure_window,
            "verified_edges": len(closure_edges),
            "first_to_last_over_path": loop_path_ratio,
            "maximum_first_to_last_over_path": 0.15,
            "first_to_last_over_median_step": loop_step_ratio,
            "maximum_first_to_last_over_median_step": maximum_loop_steps,
        },
        "trajectory_continuity": {
            "pass": trajectory_pass,
            "maximum_to_median_step_ratio": jump_ratio,
            "maximum_ratio": jump_limit,
            "raw_maximum_to_median_step_ratio": trajectory["raw_maximum_to_median_step_ratio"],
            "excluded_isolated_poses": excluded_poses,
            "maximum_excluded_isolated_poses": isolated_limit,
        },
    }
    if per_shot_registration is not None and shot_graph is not None:
        gates["per_shot_registration"] = per_shot_registration
        gates["shot_match_graph"] = shot_graph
    overall = all(
        bool(gate["pass"])
        for gate in gates.values()
        if bool(gate.get("blocking", True))
    )
    report = {
        "gate_version": GATE_VERSION,
        "profile": args.profile,
        "selected_images": selected_count,
        "registered_images": model_registered_count,
        "registration_percent": round(registration_percent, 3),
        "points": args.points,
        "mean_track_length": args.mean_track_length,
        "mean_reprojection_error_pixels": args.reprojection_error,
        "trajectory": trajectory,
        "closure_edges": closure_edges[:20],
        "shot_manifest": (
            {
                "path": str(args.shot_manifest),
                "schema_version": shot_manifest["schema_version"],
                "kind": shot_manifest["kind"],
                "candidate_frame_count": shot_manifest["candidate_frame_count"],
                "source_sha256": shot_manifest.get("source_sha256"),
                "selected_image_set_sha256": shot_manifest.get(
                    "selected_image_set_sha256"
                ),
                "shot_ids": shot_order,
            }
            if shot_manifest is not None
            else None
        ),
        "quality_gates": {**gates, "overall_pass": overall},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Pose gates {'passed' if overall else 'failed'} for {args.profile}: "
        f"{model_registered_count}/{selected_count} registered, {args.points} points, "
        f"largest match component {largest}, closure edges {len(closure_edges)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
