#!/usr/bin/env python3
"""Fail-closed mechanical and measured-quality gate for a 3DGRUT stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


GATE_VERSION = "1.0-measured-checkpoint-cap-bound"
FLOORS = {
    "smoke": {"psnr": 8.0, "ssim": 0.05, "lpips": 1.25},
    "diagnostic": {"psnr": 10.0, "ssim": 0.10, "lpips": 1.15},
    "final": {"psnr": 10.0, "ssim": 0.10, "lpips": 1.15},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_set_hash(paths: list[Path]) -> str:
    manifest = "\n".join(f"{path.name}|{sha256_file(path)}" for path in paths)
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def finite_metric(metrics: dict, name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"3DGRUT metric {name} is missing or non-numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"3DGRUT metric {name} is non-finite")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(FLOORS), required=True)
    parser.add_argument("--trainer", choices=("3DGUT", "3DGUT-MCMC"), required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--max-splats", type=int, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ply-report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--dataset-image-sha", required=True)
    parser.add_argument("--dataset-model-sha", required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--environment-sha", required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_step < 1 or args.max_splats < 10_000:
        raise ValueError("Expected step and maximum splat count must be positive")
    for value_name in (
        "source_sha",
        "dataset_image_sha",
        "dataset_model_sha",
        "config_sha",
        "environment_sha",
    ):
        if len(getattr(args, value_name)) != 64:
            raise ValueError(f"{value_name} must be a SHA256 digest")
    for required in (args.metrics, args.checkpoint, args.ply_report):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not args.renders.is_dir():
        raise FileNotFoundError(args.renders)

    metrics = json.loads(args.metrics.read_text(encoding="utf-8-sig"))
    psnr = finite_metric(metrics, "mean_psnr")
    ssim = finite_metric(metrics, "mean_ssim")
    lpips = finite_metric(metrics, "mean_lpips")
    ply_report = json.loads(args.ply_report.read_text(encoding="utf-8-sig"))
    if ply_report.get("valid_gaussian_ply") is not True:
        raise ValueError("3DGRUT PLY verifier report is not valid")
    gaussian_count = ply_report.get("vertex_count")
    if isinstance(gaussian_count, bool) or not isinstance(gaussian_count, int):
        raise ValueError("3DGRUT PLY verifier report has no integer vertex_count")
    ply_path = Path(ply_report.get("path", "")).resolve()
    if not ply_path.is_file() or sha256_file(ply_path) != ply_report.get("sha256"):
        raise ValueError("3DGRUT PLY no longer matches its verifier report")
    renders = sorted(args.renders.glob("*.png"))
    if not renders:
        raise ValueError("3DGRUT test evaluation produced no PNG renders")

    reasons: list[str] = []
    floors = FLOORS[args.stage]
    if gaussian_count > args.max_splats:
        reasons.append(
            f"Gaussian count {gaussian_count} exceeds the bound cap {args.max_splats}."
        )
    if psnr < floors["psnr"]:
        reasons.append(f"Mean PSNR {psnr:.6f} is below {floors['psnr']:.6f}.")
    if ssim < floors["ssim"]:
        reasons.append(f"Mean SSIM {ssim:.6f} is below {floors['ssim']:.6f}.")
    if lpips > floors["lpips"]:
        reasons.append(f"Mean LPIPS {lpips:.6f} exceeds {floors['lpips']:.6f}.")

    baseline_evidence = None
    if args.baseline:
        if not args.baseline.is_file():
            raise FileNotFoundError(args.baseline)
        baseline = json.loads(args.baseline.read_text(encoding="utf-8-sig"))
        if baseline.get("gate_status") != "MECHANICAL_PASS__AWAITING_VISUAL_QC":
            raise ValueError("Baseline stage was not mechanically accepted")
        baseline_metrics = baseline.get("metrics", {})
        baseline_psnr = finite_metric(baseline_metrics, "mean_psnr")
        baseline_ssim = finite_metric(baseline_metrics, "mean_ssim")
        baseline_lpips = finite_metric(baseline_metrics, "mean_lpips")
        progressed = (
            psnr >= baseline_psnr + 0.25
            or ssim >= baseline_ssim + 0.005
            or lpips <= baseline_lpips - 0.01
        )
        if not progressed:
            reasons.append("Measured quality did not improve from the preceding stage.")
        baseline_evidence = {
            "path": str(args.baseline.resolve()),
            "sha256": sha256_file(args.baseline),
            "metrics": {
                "mean_psnr": baseline_psnr,
                "mean_ssim": baseline_ssim,
                "mean_lpips": baseline_lpips,
            },
        }

    gate_status = "MECHANICAL_FAIL" if reasons else "MECHANICAL_PASS__AWAITING_VISUAL_QC"
    payload = {
        "schema_version": 1,
        "gate_version": GATE_VERSION,
        "quality_status": "rejected" if reasons else "measured_unrated",
        "gate_status": gate_status,
        "trainer": args.trainer,
        "stage": args.stage,
        "training_steps": args.expected_step,
        "max_splats": args.max_splats,
        "gaussian_count": gaussian_count,
        "metrics": {
            "mean_psnr": psnr,
            "mean_ssim": ssim,
            "mean_lpips": lpips,
        },
        "thresholds": floors,
        "reasons": reasons,
        "render_directory": str(args.renders.resolve()),
        "saved_holdout_renders": len(renders),
        "baseline": baseline_evidence,
        "binding": {
            "source_sha256": args.source_sha.lower(),
            "dataset_image_set_sha256": args.dataset_image_sha.lower(),
            "dataset_model_sha256": args.dataset_model_sha.lower(),
            "three_dgrut_config_sha256": args.config_sha.lower(),
            "three_dgrut_environment_sha256": args.environment_sha.lower(),
        },
        "input_evidence": {
            "metrics": {
                "path": str(args.metrics.resolve()),
                "sha256": sha256_file(args.metrics),
            },
            "checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": sha256_file(args.checkpoint),
            },
            "ply_report": {
                "path": str(args.ply_report.resolve()),
                "sha256": sha256_file(args.ply_report),
            },
            "ply": {
                "path": str(ply_path),
                "sha256": sha256_file(ply_path),
                "bytes": ply_path.stat().st_size,
            },
            "renders": {
                "path": str(args.renders.resolve()),
                "count": len(renders),
                "sha256": file_set_hash(renders),
            },
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"3DGRUT_STAGE_{gate_status} stage={args.stage} step={args.expected_step} "
        f"gaussians={gaussian_count}/{args.max_splats} psnr={psnr:.6f} "
        f"ssim={ssim:.6f} lpips={lpips:.6f}"
    )
    if reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
