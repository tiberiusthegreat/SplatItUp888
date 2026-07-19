#!/usr/bin/env python3
"""Select the sharpest frame from evenly spaced temporal bins."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def score_image(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        gray = image.convert("L")
        gray.thumbnail((640, 640), Image.Resampling.BILINEAR)
        pixels = np.asarray(gray, dtype=np.float32)

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

    exposure_penalty = max(0.0, 35.0 - brightness) + max(0.0, brightness - 220.0)
    contrast_penalty = max(0.0, 18.0 - contrast)
    quality = math.log1p(sharpness) - 0.035 * exposure_penalty - 0.02 * contrast_penalty
    return {
        "sharpness": round(sharpness, 4),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "quality": round(quality, 6),
    }


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
        draw.rectangle((x, y + tile_height - 20, x + 132, y + tile_height), fill="black")
        draw.text((x + 5, y + tile_height - 17), path.stem, fill="white")

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=180)
    parser.add_argument("--blur-percentile", type=float, default=20.0)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    args = parser.parse_args()

    frames = sorted(args.input.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"No JPEG frames found in {args.input}")
    if args.target < 2:
        raise SystemExit("--target must be at least 2")

    target = min(args.target, len(frames))
    scored = []
    for index, path in enumerate(frames):
        record = {"index": index, "filename": path.name, **score_image(path)}
        scored.append(record)

    blur_threshold = float(
        np.percentile([record["sharpness"] for record in scored], args.blur_percentile)
    )
    for record in scored:
        record["below_blur_percentile"] = record["sharpness"] < blur_threshold

    selected_indices: set[int] = set()
    for bin_index in range(target):
        start = math.floor(bin_index * len(frames) / target)
        end = math.floor((bin_index + 1) * len(frames) / target)
        candidates = scored[start : max(start + 1, end)]
        sharp_candidates = [item for item in candidates if not item["below_blur_percentile"]]
        winner = max(sharp_candidates or candidates, key=lambda item: item["quality"])
        selected_indices.add(int(winner["index"]))

    selected = [record for record in scored if record["index"] in selected_indices]
    args.output.mkdir(parents=True, exist_ok=True)
    for old_file in args.output.glob("*.jpg"):
        old_file.unlink()
    for record in selected:
        shutil.copy2(args.input / record["filename"], args.output / record["filename"])
        record["selected"] = True
    for record in scored:
        record.setdefault("selected", False)

    args.records.parent.mkdir(parents=True, exist_ok=True)
    args.records.write_text(
        json.dumps(
            {
                "source_frames": len(frames),
                "requested_frames": args.target,
                "selected_frames": len(selected),
                "blur_percentile": args.blur_percentile,
                "blur_threshold": round(blur_threshold, 4),
                "globally_blurry_frames": sum(
                    1 for record in scored if record["below_blur_percentile"]
                ),
                "records": scored,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    csv_path = args.records.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scored[0].keys())
        writer.writeheader()
        writer.writerows(scored)

    selected_paths = [args.output / record["filename"] for record in selected]
    build_contact_sheet(selected_paths, args.contact_sheet)
    print(
        f"Selected {len(selected)} of {len(frames)} frames; "
        f"avoided the blurriest {args.blur_percentile:g}% when each time bin allowed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
