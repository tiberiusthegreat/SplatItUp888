#!/usr/bin/env python3
"""Build a deterministic, capped COLMAP initialization for a 3DGRUT smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


SCHEMA_VERSION = 1
SAMPLER_VERSION = "reservoir-seed-42-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_points(source: Path, destination: Path, maximum: int) -> tuple[int, int]:
    comments: list[str] = []
    reservoir: list[tuple[int, str]] = []
    rng = random.Random(42)
    point_count = 0

    with source.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                comments.append(line)
                continue
            if len(line.split()) < 8:
                raise ValueError(f"Malformed COLMAP point record at source row {point_count + 1}")
            if point_count < maximum:
                reservoir.append((point_count, line))
            else:
                replacement = rng.randrange(point_count + 1)
                if replacement < maximum:
                    reservoir[replacement] = (point_count, line)
            point_count += 1

    if point_count < 1:
        raise ValueError("COLMAP points3D.txt contains no points")
    reservoir.sort(key=lambda item: item[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for line in comments:
            stream.write(line.rstrip("\r\n") + "\n")
        for _, line in reservoir:
            stream.write(line.rstrip("\r\n") + "\n")
    return point_count, len(reservoir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_points < 10_000:
        raise ValueError("--max-points must be at least 10000")
    source_sparse = args.source_dataset / "sparse" / "0"
    source_cameras = source_sparse / "cameras.bin"
    source_images = source_sparse / "images.bin"
    for required in (source_cameras, source_images, args.source_points):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Smoke dataset output is not empty: {args.output}")

    output_sparse = args.output / "sparse" / "0"
    output_sparse.mkdir(parents=True, exist_ok=True)
    output_cameras = output_sparse / "cameras.bin"
    output_images = output_sparse / "images.bin"
    output_points = output_sparse / "points3D.txt"
    shutil.copy2(source_cameras, output_cameras)
    shutil.copy2(source_images, output_images)
    input_points, output_point_count = sample_points(
        args.source_points, output_points, args.max_points
    )
    if output_point_count > args.max_points:
        raise AssertionError("Smoke point sampler exceeded its cap")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "source_dataset": str(args.source_dataset.resolve()),
        "source_points": {
            "path": str(args.source_points.resolve()),
            "sha256": sha256_file(args.source_points),
            "count": input_points,
        },
        "max_points": args.max_points,
        "output_dataset": str(args.output.resolve()),
        "output": {
            "cameras_sha256": sha256_file(output_cameras),
            "images_sha256": sha256_file(output_images),
            "points_sha256": sha256_file(output_points),
            "point_count": output_point_count,
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"3DGRUT_SMOKE_DATASET_READY points={output_point_count}/{input_points} "
        f"cap={args.max_points}"
    )


if __name__ == "__main__":
    main()
