#!/usr/bin/env python3
"""Measure deterministic Brush holdout renders against their source images."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


EVALUATOR_VERSION = "1.3-brush-render-name-contract"


def read_registered_names(path: Path) -> set[str]:
    names: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=9)
        if len(parts) < 10:
            raise SystemExit(f"Malformed COLMAP image line: {line}")
        names.add(parts[9])
        if index < len(lines):
            index += 1
    return names


def reference_name_for_render(path: Path) -> str:
    nested_name = path.stem
    if Path(nested_name).suffix.casefold() in {".jpg", ".jpeg"}:
        return nested_name
    return f"{nested_name}.jpg"


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image.astype(np.float32) / 255.0


def ssim(first: np.ndarray, second: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_first = cv2.GaussianBlur(first, (11, 11), 1.5)
    mu_second = cv2.GaussianBlur(second, (11, 11), 1.5)
    mu_first_sq = mu_first * mu_first
    mu_second_sq = mu_second * mu_second
    mu_cross = mu_first * mu_second
    sigma_first = cv2.GaussianBlur(first * first, (11, 11), 1.5) - mu_first_sq
    sigma_second = cv2.GaussianBlur(second * second, (11, 11), 1.5) - mu_second_sq
    sigma_cross = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mu_cross
    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_first_sq + mu_second_sq + c1) * (sigma_first + sigma_second + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--reference-images", type=Path, required=True)
    parser.add_argument("--registered-images", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    renders = sorted(args.renders.glob("*.png"))
    if len(renders) != args.expected:
        raise SystemExit(f"Found {len(renders)} holdout renders; expected exactly {args.expected}")
    registered_names = read_registered_names(args.registered_images)
    unregistered = [
        reference_name_for_render(render)
        for render in renders
        if reference_name_for_render(render) not in registered_names
    ]
    if unregistered:
        raise SystemExit(
            "Holdout renders used cameras absent from the cleaned COLMAP model: "
            + ", ".join(unregistered)
        )

    records = []
    for render_path in renders:
        reference_path = args.reference_images / reference_name_for_render(render_path)
        if not reference_path.is_file():
            raise SystemExit(f"Missing holdout reference image: {reference_path}")
        rendered = read_color(render_path)
        reference = read_color(reference_path)
        if reference.shape[:2] != rendered.shape[:2]:
            reference = cv2.resize(
                reference,
                (rendered.shape[1], rendered.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        mse = float(np.mean((rendered - reference) ** 2))
        psnr = 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
        image_ssim = ssim(rendered, reference)
        if not math.isfinite(psnr) or not math.isfinite(image_ssim):
            raise SystemExit(f"Non-finite holdout metric for {render_path.name}")
        records.append(
            {
                "image": render_path.name,
                "reference": reference_path.name,
                "width": rendered.shape[1],
                "height": rendered.shape[0],
                "psnr": round(psnr, 6),
                "ssim": round(image_ssim, 8),
            }
        )

    psnr_values = np.asarray([record["psnr"] for record in records], dtype=np.float64)
    ssim_values = np.asarray([record["ssim"] for record in records], dtype=np.float64)
    report = {
        "evaluator_version": EVALUATOR_VERSION,
        "evaluation": "photometric holdout; COLMAP geometry still used all registered frames",
        "quality_status": "measured_unrated",
        "render_directory": str(args.renders.resolve()),
        "reference_directory": str(args.reference_images.resolve()),
        "registered_images_file": str(args.registered_images.resolve()),
        "expected_holdout_renders": args.expected,
        "saved_holdout_renders": len(records),
        "mean_psnr": round(float(np.mean(psnr_values)), 6),
        "median_psnr": round(float(np.median(psnr_values)), 6),
        "mean_ssim": round(float(np.mean(ssim_values)), 8),
        "median_ssim": round(float(np.median(ssim_values)), 8),
        "absolute_quality_threshold": None,
        "note": "Metrics are recorded for repeatable A/B comparisons; no unsupported absolute visual-quality threshold is imposed.",
        "images": records,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Measured {len(records)} holdout renders: "
        f"mean PSNR {report['mean_psnr']:.3f}, mean SSIM {report['mean_ssim']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
