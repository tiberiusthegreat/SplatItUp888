#!/usr/bin/env python3
"""Independently verify Spirula's saved holdout pairs and native metrics."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np


EVALUATOR_VERSION = "1.1-spirula-paired-holdout-deduplicated"


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def finite_metric_list(payload: dict[str, object], name: str, expected: int) -> list[float]:
    values = payload.get(name)
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError(f"Native {name} must contain exactly {expected} values")
    return [finite_number(value, f"{name}[{index}]") for index, value in enumerate(values)]


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read holdout image: {path}")
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


def paired_files(run_dir: Path, expected: int) -> tuple[list[tuple[Path, Path]], bool]:
    normalized_references = sorted((run_dir / "holdout_references").glob("eval-*.png"))
    normalized_renders = sorted((run_dir / "holdout_renders").glob("eval-*.png"))
    if len(normalized_references) == expected and len(normalized_renders) == expected:
        return list(zip(normalized_references, normalized_renders, strict=True)), True

    ground_truth = sorted(run_dir.glob("eval-gt-*.png"))
    renders = sorted(run_dir.glob("eval-render-*.png"))
    if len(ground_truth) != expected or len(renders) != expected:
        raise ValueError(
            f"Spirula saved {len(ground_truth)} ground-truth and {len(renders)} render images; "
            f"expected exactly {expected} of each"
        )
    pairs = []
    for gt_path, render_path in zip(ground_truth, renders, strict=True):
        if gt_path.name.removeprefix("eval-gt-") != render_path.name.removeprefix("eval-render-"):
            raise ValueError(f"Unpaired Spirula holdout files: {gt_path.name}, {render_path.name}")
        pairs.append((gt_path, render_path))
    return pairs, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    if args.expected < 1:
        raise SystemExit("Expected holdout count must be positive")
    metrics_path = args.run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise SystemExit(f"Spirula metrics are missing: {metrics_path}")

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise ValueError("Spirula metrics root must be an object")
        if metrics.get("num_eval_images") != args.expected:
            raise ValueError(
                f"Native num_eval_images is {metrics.get('num_eval_images')}; expected {args.expected}"
            )
        native_psnr = finite_metric_list(metrics, "psnr", args.expected)
        native_ssim = finite_metric_list(metrics, "ssim", args.expected)
        native_mean_psnr = finite_number(metrics.get("avg_psnr"), "avg_psnr")
        native_mean_ssim = finite_number(metrics.get("avg_ssim"), "avg_ssim")
        training_time = finite_number(metrics.get("training_time"), "training_time")
        engine_vram = finite_number(metrics.get("engine_vram"), "engine_vram")
        pairs, already_normalized = paired_files(args.run_dir, args.expected)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    render_directory = args.run_dir / "holdout_renders"
    reference_directory = args.run_dir / "holdout_references"
    if not already_normalized:
        for directory in (render_directory, reference_directory):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True)

    records = []
    for index, (gt_path, render_path) in enumerate(pairs):
        ground_truth = read_color(gt_path)
        rendered = read_color(render_path)
        if ground_truth.shape != rendered.shape:
            raise SystemExit(f"Holdout pair dimensions differ: {gt_path.name}, {render_path.name}")
        mse = float(np.mean((rendered - ground_truth) ** 2))
        psnr = 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
        image_ssim = ssim(rendered, ground_truth)
        if not math.isfinite(psnr) or not math.isfinite(image_ssim):
            raise SystemExit(f"Non-finite independent metric for pair {index}")
        normalized_name = f"eval-{index:05d}.png"
        if not already_normalized:
            shutil.copy2(render_path, render_directory / normalized_name)
            shutil.copy2(gt_path, reference_directory / normalized_name)
        records.append(
            {
                "image": normalized_name,
                "source_render": render_path.name,
                "source_reference": gt_path.name,
                "width": rendered.shape[1],
                "height": rendered.shape[0],
                "psnr": round(psnr, 6),
                "ssim": round(image_ssim, 8),
                "native_psnr": round(native_psnr[index], 6),
                "native_ssim": round(native_ssim[index], 8),
            }
        )

    psnr_values = np.asarray([record["psnr"] for record in records], dtype=np.float64)
    ssim_values = np.asarray([record["ssim"] for record in records], dtype=np.float64)
    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))
    if abs(mean_psnr - native_mean_psnr) > 0.1:
        raise SystemExit(
            f"Independent mean PSNR {mean_psnr:.6f} disagrees with Spirula {native_mean_psnr:.6f}"
        )
    if abs(mean_ssim - native_mean_ssim) > 0.005:
        raise SystemExit(
            f"Independent mean SSIM {mean_ssim:.8f} disagrees with Spirula {native_mean_ssim:.8f}"
        )

    report = {
        "evaluator_version": EVALUATOR_VERSION,
        "trainer": "Spirula",
        "evaluation": "paired Spirula holdout; independently rescored from saved ground truth and render PNGs",
        "quality_status": "measured_unrated",
        "render_directory": str(render_directory.resolve()),
        "reference_directory": str(reference_directory.resolve()),
        "native_metrics_path": str(metrics_path.resolve()),
        "expected_holdout_renders": args.expected,
        "saved_holdout_renders": len(records),
        "mean_psnr": round(mean_psnr, 6),
        "median_psnr": round(float(np.median(psnr_values)), 6),
        "mean_ssim": round(mean_ssim, 8),
        "median_ssim": round(float(np.median(ssim_values)), 8),
        "native_mean_psnr": round(native_mean_psnr, 6),
        "native_mean_ssim": round(native_mean_ssim, 8),
        "training_time_seconds": training_time,
        "engine_vram_mb": engine_vram,
        "absolute_quality_threshold": None,
        "note": "Metrics are retained for controlled A/B comparison; visual approval remains required.",
        "images": records,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not already_normalized:
        for gt_path, render_path in pairs:
            gt_path.unlink()
            render_path.unlink()
    print(
        f"Verified {len(records)} Spirula holdouts: "
        f"mean PSNR {mean_psnr:.3f}, mean SSIM {mean_ssim:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
