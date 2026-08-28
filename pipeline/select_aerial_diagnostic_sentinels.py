#!/usr/bin/env python3
"""Select deterministic close and control holdouts from a COLMAP text model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


ALGORITHM_VERSION = "1.0-colmap-distance-bands-temporal-spacing"
CLOSE_COUNT = 8
CONTROL_COUNT = 4
MIN_TRACKED_OBSERVATIONS = 20
CANDIDATE_BAND_FRACTION = 0.40
MIN_CONTROL_TO_CLOSE_MEDIAN_RATIO = 1.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_floats(values: list[str], label: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(value) for value in values)
    except ValueError as exc:
        raise ValueError(f"Malformed {label}: expected numeric values") from exc
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"Malformed {label}: values must be finite")
    return parsed


def read_cameras(path: Path) -> dict[int, dict[str, object]]:
    cameras: dict[int, dict[str, object]] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed COLMAP camera line: {line}")
        try:
            camera_id = int(parts[0])
            width = int(parts[2])
            height = int(parts[3])
        except ValueError as exc:
            raise ValueError(f"Malformed COLMAP camera line: {line}") from exc
        if camera_id < 1 or width < 1 or height < 1:
            raise ValueError(f"Invalid COLMAP camera values: {line}")
        if camera_id in cameras:
            raise ValueError(f"Duplicate COLMAP camera ID: {camera_id}")
        parameters = _finite_floats(parts[4:], f"camera {camera_id}")
        cameras[camera_id] = {
            "camera_id": camera_id,
            "model": parts[1],
            "width": width,
            "height": height,
            "parameters": parameters,
        }
    if not cameras:
        raise ValueError("COLMAP cameras.txt contains no cameras")
    return cameras


def quaternion_rotation(qvec: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    norm = math.sqrt(sum(value * value for value in qvec))
    if norm == 0.0:
        raise ValueError("COLMAP image quaternion must be nonzero")
    qw, qx, qy, qz = (value / norm for value in qvec)
    return (
        (
            1 - 2 * qy * qy - 2 * qz * qz,
            2 * qx * qy - 2 * qw * qz,
            2 * qx * qz + 2 * qw * qy,
        ),
        (
            2 * qx * qy + 2 * qw * qz,
            1 - 2 * qx * qx - 2 * qz * qz,
            2 * qy * qz - 2 * qw * qx,
        ),
        (
            2 * qx * qz - 2 * qw * qy,
            2 * qy * qz + 2 * qw * qx,
            1 - 2 * qx * qx - 2 * qy * qy,
        ),
    )


def camera_center(
    qvec: tuple[float, ...], tvec: tuple[float, ...]
) -> tuple[float, float, float]:
    rotation = quaternion_rotation(qvec)
    return tuple(
        -sum(rotation[row][column] * tvec[row] for row in range(3))
        for column in range(3)
    )


def _observation_point_ids(line: str, image_id: int) -> tuple[int, ...]:
    if not line:
        return ()
    parts = line.split()
    if len(parts) % 3:
        raise ValueError(f"Malformed POINTS2D record for image {image_id}")
    point_ids: list[int] = []
    for index in range(0, len(parts), 3):
        _finite_floats(parts[index : index + 2], f"POINTS2D image {image_id}")
        try:
            point_id = int(parts[index + 2])
        except ValueError as exc:
            raise ValueError(
                f"Malformed POINT3D_ID in image {image_id}"
            ) from exc
        if point_id < -1:
            raise ValueError(f"Invalid POINT3D_ID in image {image_id}: {point_id}")
        if point_id >= 0:
            point_ids.append(point_id)
    return tuple(point_ids)


def read_images(path: Path) -> dict[int, dict[str, object]]:
    images: dict[int, dict[str, object]] = {}
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        header = lines[index].strip()
        index += 1
        if not header or header.startswith("#"):
            continue
        parts = header.split(maxsplit=9)
        if len(parts) != 10:
            raise ValueError(f"Malformed COLMAP image line: {header}")
        try:
            image_id = int(parts[0])
            camera_id = int(parts[8])
        except ValueError as exc:
            raise ValueError(f"Malformed COLMAP image line: {header}") from exc
        if image_id < 1 or camera_id < 1 or not parts[9]:
            raise ValueError(f"Invalid COLMAP image values: {header}")
        if image_id in images:
            raise ValueError(f"Duplicate COLMAP image ID: {image_id}")
        qvec = _finite_floats(parts[1:5], f"image {image_id} quaternion")
        tvec = _finite_floats(parts[5:8], f"image {image_id} translation")

        while index < len(lines) and lines[index].lstrip().startswith("#"):
            index += 1
        observations = lines[index].strip() if index < len(lines) else ""
        if index < len(lines):
            index += 1
        images[image_id] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "name": parts[9],
            "center": camera_center(qvec, tvec),
            # Parse POINTS2D only for prospective holdouts after IMAGE_ID ordering.
            # Dense text models can contain millions of non-holdout observations.
            "observations": observations,
        }
    if not images:
        raise ValueError("COLMAP images.txt contains no registered images")
    return images


def read_points3d(path: Path) -> dict[int, tuple[float, float, float]]:
    points: dict[int, tuple[float, float, float]] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=8)
        if len(parts) < 8:
            raise ValueError(f"Malformed COLMAP point line: {line}")
        try:
            point_id = int(parts[0])
            tuple(int(value) for value in parts[4:7])
        except ValueError as exc:
            raise ValueError(f"Malformed COLMAP point line: {line}") from exc
        if point_id < 0 or point_id in points:
            raise ValueError(f"Invalid or duplicate COLMAP point ID: {point_id}")
        xyz = _finite_floats(parts[1:4], f"point {point_id} position")
        _finite_floats([parts[7]], f"point {point_id} error")
        points[point_id] = (xyz[0], xyz[1], xyz[2])
    if not points:
        raise ValueError("COLMAP points3D.txt contains no sparse points")
    return points


def temporally_spaced(records: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    chronological = sorted(
        records,
        key=lambda item: (
            int(item["registered_position"]),
            int(item["image_id"]),
            str(item["name"]),
        ),
    )
    if len(chronological) < count:
        raise ValueError(f"Need {count} candidates, found {len(chronological)}")
    if count == 1:
        return [chronological[len(chronological) // 2]]
    denominator = count - 1
    last_index = len(chronological) - 1
    indices = [
        (2 * ordinal * last_index + denominator) // (2 * denominator)
        for ordinal in range(count)
    ]
    return [chronological[index] for index in indices]


def select_sentinels(model_text: Path, eval_split_every: int) -> dict[str, object]:
    if eval_split_every < 1:
        raise ValueError("eval-split-every must be a positive integer")
    model_text = model_text.resolve()
    paths = {
        name: model_text / name
        for name in ("cameras.txt", "images.txt", "points3D.txt")
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Missing COLMAP text file(s): {', '.join(missing)}")

    cameras = read_cameras(paths["cameras.txt"])
    images = read_images(paths["images.txt"])
    points = read_points3d(paths["points3D.txt"])
    unknown_cameras = sorted(
        {
            int(image["camera_id"])
            for image in images.values()
            if int(image["camera_id"]) not in cameras
        }
    )
    if unknown_cameras:
        raise ValueError(
            "Registered images reference missing camera ID(s): "
            + ", ".join(str(value) for value in unknown_cameras)
        )

    registered = [images[image_id] for image_id in sorted(images)]
    holdouts: list[dict[str, object]] = []
    for holdout_ordinal, registered_index in enumerate(
        range(0, len(registered), eval_split_every), start=1
    ):
        image = registered[registered_index]
        observed_ids = sorted(
            set(
                _observation_point_ids(
                    str(image["observations"]), int(image["image_id"])
                )
            )
        )
        missing_points = [point_id for point_id in observed_ids if point_id not in points]
        if missing_points:
            raise ValueError(
                f"Image {image['image_id']} references missing sparse point ID(s): "
                + ", ".join(str(value) for value in missing_points[:10])
            )
        center = tuple(float(value) for value in image["center"])
        distances = [math.dist(center, points[point_id]) for point_id in observed_ids]
        if not all(math.isfinite(value) and value > 0.0 for value in distances):
            raise ValueError(f"Image {image['image_id']} has invalid camera-to-point distance")
        median_distance = float(statistics.median(distances)) if distances else None
        tracked_count = len(distances)
        holdouts.append(
            {
                "holdout_ordinal": holdout_ordinal,
                "registered_position": registered_index + 1,
                "image_id": int(image["image_id"]),
                "name": str(image["name"]),
                "tracked_observation_count": tracked_count,
                "median_camera_to_observed_point_distance": median_distance,
                "eligible": tracked_count >= MIN_TRACKED_OBSERVATIONS,
            }
        )

    eligible = [item for item in holdouts if item["eligible"]]
    minimum_eligible = math.ceil(CLOSE_COUNT / CANDIDATE_BAND_FRACTION)
    if len(eligible) < minimum_eligible:
        raise ValueError(
            f"Need at least {minimum_eligible} eligible prospective holdouts with "
            f"at least {MIN_TRACKED_OBSERVATIONS} tracked observations; found {len(eligible)}"
        )
    ranked = sorted(
        eligible,
        key=lambda item: (
            float(item["median_camera_to_observed_point_distance"]),
            int(item["registered_position"]),
            int(item["image_id"]),
            str(item["name"]),
        ),
    )
    band_count = math.floor(len(ranked) * CANDIDATE_BAND_FRACTION)
    close_candidates = ranked[:band_count]
    control_candidates = ranked[-band_count:]
    if len(close_candidates) < CLOSE_COUNT or len(control_candidates) < CONTROL_COUNT:
        raise ValueError("Insufficient low-distance or high-distance candidates")
    close_band_max = max(
        float(item["median_camera_to_observed_point_distance"])
        for item in close_candidates
    )
    control_band_min = min(
        float(item["median_camera_to_observed_point_distance"])
        for item in control_candidates
    )
    if close_band_max >= control_band_min:
        raise ValueError(
            "Ambiguous close/control distance bands: strict separation is required"
        )

    selected_close = temporally_spaced(close_candidates, CLOSE_COUNT)
    selected_controls = temporally_spaced(control_candidates, CONTROL_COUNT)
    close_median = float(
        statistics.median(
            float(item["median_camera_to_observed_point_distance"])
            for item in selected_close
        )
    )
    control_median = float(
        statistics.median(
            float(item["median_camera_to_observed_point_distance"])
            for item in selected_controls
        )
    )
    distance_ratio = control_median / close_median
    if distance_ratio < MIN_CONTROL_TO_CLOSE_MEDIAN_RATIO:
        raise ValueError(
            f"Ambiguous close/control distance separation: ratio {distance_ratio:.6f} "
            f"is below {MIN_CONTROL_TO_CLOSE_MEDIAN_RATIO:.6f}"
        )

    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "selection_status": "MECHANICAL_SELECTION",
        "parameters": {
            "eval_split_every": eval_split_every,
            "registered_image_order": "numeric_image_id_ascending",
            "prospective_holdout_positions": "1,1+N,1+2N,...",
            "minimum_tracked_observations_per_holdout": MIN_TRACKED_OBSERVATIONS,
            "candidate_band_fraction": CANDIDATE_BAND_FRACTION,
            "close_selection_count": CLOSE_COUNT,
            "control_selection_count": CONTROL_COUNT,
            "minimum_control_to_close_median_distance_ratio": MIN_CONTROL_TO_CLOSE_MEDIAN_RATIO,
            "strict_candidate_band_nonoverlap": True,
        },
        "source_files": {name: str(path) for name, path in paths.items()},
        "source_hashes": source_hashes,
        "counts": {
            "cameras": len(cameras),
            "registered_images": len(registered),
            "sparse_points": len(points),
            "prospective_holdouts": len(holdouts),
            "eligible_holdouts": len(eligible),
            "ineligible_holdouts": len(holdouts) - len(eligible),
            "close_candidates": len(close_candidates),
            "control_candidates": len(control_candidates),
            "selected_close": len(selected_close),
            "selected_controls": len(selected_controls),
        },
        "distance_separation": {
            "close_candidate_max": close_band_max,
            "control_candidate_min": control_band_min,
            "selected_close_median": close_median,
            "selected_control_median": control_median,
            "selected_control_to_close_median_ratio": distance_ratio,
        },
        "close_images": [str(item["name"]) for item in selected_close],
        "control_images": [str(item["name"]) for item in selected_controls],
        "selected_close": selected_close,
        "selected_controls": selected_controls,
        "close_candidates": close_candidates,
        "control_candidates": control_candidates,
        "prospective_holdouts": holdouts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-text", type=Path, required=True)
    parser.add_argument("--eval-split-every", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = select_sentinels(args.model_text, args.eval_split_every)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        "AERIAL_DIAGNOSTIC_SENTINELS_SELECTED "
        f"close={len(report['close_images'])} controls={len(report['control_images'])} "
        f"ratio={report['distance_separation']['selected_control_to_close_median_ratio']:.6f}"
    )


if __name__ == "__main__":
    main()
