#!/usr/bin/env python3
"""Apply a fail-closed quality gate to an aerial-exterior holdout report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


GATE_VERSION = "1.5-topology-aware-input-bound-fail-closed"
MIN_SETTLING_STEPS = 2000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_input_evidence(
    report_path: Path,
    report_bytes: bytes,
    ply_report_path: Path,
    ply_report_bytes: bytes,
    ply_report: dict,
) -> dict:
    if not report_path.is_file() or not ply_report_path.is_file():
        raise ValueError("Gate input reports must exist as files")
    ply_path_value = ply_report.get("path")
    reported_ply_sha = ply_report.get("sha256")
    if not isinstance(ply_path_value, str) or not ply_path_value:
        raise ValueError("PLY report path is required")
    if not isinstance(reported_ply_sha, str) or len(reported_ply_sha) != 64:
        raise ValueError("PLY report SHA256 is required")
    ply_path = Path(ply_path_value).resolve()
    if not ply_path.is_file():
        raise ValueError(f"Gated PLY is missing: {ply_path}")
    actual_ply_sha = sha256_file(ply_path)
    if actual_ply_sha != reported_ply_sha.lower():
        raise ValueError("Gated PLY does not match its verifier report")
    return {
        "quality_report": {
            "path": str(report_path.resolve()),
            "sha256": hashlib.sha256(report_bytes).hexdigest(),
        },
        "ply_report": {
            "path": str(ply_report_path.resolve()),
            "sha256": hashlib.sha256(ply_report_bytes).hexdigest(),
        },
        "ply": {
            "path": str(ply_path),
            "bytes": ply_path.stat().st_size,
            "sha256": actual_ply_sha,
        },
    }


def image_stem(value: str) -> str:
    path = Path(value)
    stem = path.stem
    if path.suffix.casefold() == ".png" and Path(stem).suffix.casefold() in {".jpg", ".jpeg"}:
        return Path(stem).stem
    return stem


def validate_report_contract(report: dict) -> None:
    if not isinstance(report, dict):
        raise ValueError("Holdout report must be a JSON object")
    if report.get("quality_status") != "measured_unrated":
        raise ValueError("Holdout report quality_status must be measured_unrated")
    evaluator_version = report.get("evaluator_version")
    if not isinstance(evaluator_version, str) or not evaluator_version:
        raise ValueError("Holdout report evaluator_version is required")
    images = report.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Holdout report must contain at least one image metric")
    expected = report.get("expected_holdout_renders")
    saved = report.get("saved_holdout_renders")
    if expected != len(images) or saved != len(images):
        raise ValueError("Holdout report cardinality does not match its image metrics")

    stems: list[str] = []
    for item in images:
        if not isinstance(item, dict) or not isinstance(item.get("image"), str):
            raise ValueError("Every holdout metric must have an image filename")
        stem = image_stem(item["image"])
        stems.append(stem)
        for metric in ("ssim", "psnr"):
            value = item.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"Non-finite or missing {metric} for holdout {item['image']}")
        if not 0.0 <= item["ssim"] <= 1.0:
            raise ValueError(f"SSIM is outside [0, 1] for holdout {item['image']}")
        if not 0.0 <= item["psnr"] <= 100.0:
            raise ValueError(f"PSNR is outside [0, 100] for holdout {item['image']}")
    if len(stems) != len(set(stems)):
        raise ValueError("Holdout report contains duplicate image stems")


def select_images(report: dict, requested: list[str], label: str) -> list[dict]:
    if not requested:
        raise ValueError(f"At least one {label} holdout image is required")
    by_stem = {image_stem(item["image"]): item for item in report.get("images", [])}
    stems = [image_stem(value) for value in requested]
    if len(stems) != len(set(stems)):
        raise ValueError(f"Duplicate {label} holdout image requests are not allowed")
    missing = [stem for stem in stems if stem not in by_stem]
    if missing:
        raise ValueError(f"Missing {label} holdout image(s): {', '.join(missing)}")
    selected = [by_stem[stem] for stem in stems]
    return selected


def evaluate_gate(
    report: dict,
    close_names: list[str],
    control_names: list[str],
    min_control_median_ssim: float = 0.60,
    min_close_median_ssim: float = 0.52,
    close_control_ratio: float = 0.80,
    min_close_median_psnr: float = 22.0,
    low_close_ssim_threshold: float = 0.45,
    max_low_close_images: int = 2,
    training_steps: int | None = None,
    growth_stop_iter: int | None = None,
    gaussian_count: int | None = None,
    max_splats: int | None = None,
    last_topology_change_iter: int | None = None,
) -> dict:
    validate_report_contract(report)
    if training_steps is None or growth_stop_iter is None:
        raise ValueError("Training steps and growth-stop iteration are required")
    if gaussian_count is None or max_splats is None:
        raise ValueError("Gaussian count and maximum splat count are required")
    if last_topology_change_iter is None:
        last_topology_change_iter = growth_stop_iter
    state_values = (
        training_steps,
        growth_stop_iter,
        gaussian_count,
        max_splats,
        last_topology_change_iter,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in state_values):
        raise ValueError("Training and splat-count state must contain integers")
    if (
        training_steps < 1
        or growth_stop_iter < 1
        or max_splats < 1
        or gaussian_count < 0
        or last_topology_change_iter < 0
    ):
        raise ValueError("Training and splat-count state must contain positive limits")
    if gaussian_count > max_splats:
        raise ValueError("Gaussian count exceeds the configured maximum splat count")

    close_stems = [image_stem(value) for value in close_names]
    control_stems = [image_stem(value) for value in control_names]
    overlap = sorted(set(close_stems) & set(control_stems))
    if overlap:
        raise ValueError(
            f"Close and control holdouts must be disjoint: {', '.join(overlap)}"
        )

    close = select_images(report, close_names, "close")
    controls = select_images(report, control_names, "control")

    close_median_ssim = statistics.median(item["ssim"] for item in close)
    close_median_psnr = statistics.median(item["psnr"] for item in close)
    control_median_ssim = statistics.median(item["ssim"] for item in controls)
    required_close_ssim = max(
        min_close_median_ssim,
        close_control_ratio * control_median_ssim,
    )
    low_close_count = sum(
        item["ssim"] < low_close_ssim_threshold for item in close
    )

    reasons: list[str] = []
    cap_bound = (
        gaussian_count is not None
        and max_splats is not None
        and gaussian_count >= 0.995 * max_splats
    )
    growth_active = training_steps < growth_stop_iter
    topology_active = training_steps < last_topology_change_iter
    settling_steps = training_steps - last_topology_change_iter

    if cap_bound:
        status = "INCONCLUSIVE_CAP_BOUND"
        reasons.append(
            f"Gaussian count {gaussian_count} reached the diagnostic cap of {max_splats}."
        )
    elif growth_active or topology_active:
        status = "INCONCLUSIVE_GROWTH_ACTIVE"
        reasons.append(
            f"Training stopped at {training_steps} before growth/topology changes end at "
            f"{max(growth_stop_iter, last_topology_change_iter)}; a mature short diagnostic "
            "is still required."
        )
    elif settling_steps < MIN_SETTLING_STEPS:
        status = "INCONCLUSIVE_SETTLING"
        reasons.append(
            f"Only {settling_steps} post-growth settling steps completed; at least "
            f"{MIN_SETTLING_STEPS} are required for a mature diagnostic."
        )
    elif control_median_ssim < min_control_median_ssim:
        status = "INCONCLUSIVE_CONTROLS"
        reasons.append(
            "Stable control views are not mature enough to judge close-view quality."
        )
    else:
        if close_median_ssim < required_close_ssim:
            reasons.append(
                f"Close median SSIM {close_median_ssim:.6f} is below "
                f"the required {required_close_ssim:.6f}."
            )
        if close_median_psnr < min_close_median_psnr:
            reasons.append(
                f"Close median PSNR {close_median_psnr:.6f} dB is below "
                f"the required {min_close_median_psnr:.6f} dB."
            )
        if low_close_count > max_low_close_images:
            reasons.append(
                f"{low_close_count} close views are below SSIM "
                f"{low_close_ssim_threshold:.6f}; at most {max_low_close_images} are allowed."
            )
        if not reasons:
            status = "MECHANICAL_PASS__AWAITING_VISUAL_QC"
        else:
            status = "MECHANICAL_FAIL"

    return {
        "gate_version": GATE_VERSION,
        "quality_status": status,
        "metrics": {
            "close_median_ssim": close_median_ssim,
            "close_median_psnr": close_median_psnr,
            "control_median_ssim": control_median_ssim,
            "required_close_median_ssim": required_close_ssim,
            "low_close_ssim_count": low_close_count,
        },
        "thresholds": {
            "min_control_median_ssim": min_control_median_ssim,
            "min_close_median_ssim": min_close_median_ssim,
            "close_control_ratio": close_control_ratio,
            "min_close_median_psnr": min_close_median_psnr,
            "low_close_ssim_threshold": low_close_ssim_threshold,
            "max_low_close_images": max_low_close_images,
        },
        "training_state": {
            "training_steps": training_steps,
            "growth_stop_iter": growth_stop_iter,
            "growth_active": growth_active,
            "last_topology_change_iter": last_topology_change_iter,
            "topology_active": topology_active,
            "settling_steps": settling_steps,
            "minimum_settling_steps": MIN_SETTLING_STEPS,
            "gaussian_count": gaussian_count,
            "max_splats": max_splats,
            "cap_bound": cap_bound,
        },
        "close_images": close,
        "control_images": controls,
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--close-images", nargs="+", required=True)
    parser.add_argument("--control-images", nargs="+", required=True)
    parser.add_argument("--training-steps", type=int, required=True)
    parser.add_argument("--growth-stop-iter", type=int, required=True)
    parser.add_argument("--last-topology-change-iter", type=int)
    parser.add_argument("--max-splats", type=int, required=True)
    parser.add_argument("--ply-report", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_bytes = args.report.read_bytes()
    ply_report_bytes = args.ply_report.read_bytes()
    report = json.loads(report_bytes.decode("utf-8-sig"))
    ply_report = json.loads(ply_report_bytes.decode("utf-8-sig"))
    if ply_report.get("valid_gaussian_ply") is not True:
        raise ValueError("PLY report does not describe a valid Gaussian PLY")
    if ply_report.get("nonfinite_vertices") != 0:
        raise ValueError("PLY report contains non-finite vertices")
    gaussian_count = ply_report.get("vertex_count")
    result = evaluate_gate(
        report,
        args.close_images,
        args.control_images,
        training_steps=args.training_steps,
        growth_stop_iter=args.growth_stop_iter,
        gaussian_count=gaussian_count,
        max_splats=args.max_splats,
        last_topology_change_iter=args.last_topology_change_iter,
    )
    result["input_evidence"] = build_input_evidence(
        args.report,
        report_bytes,
        args.ply_report,
        ply_report_bytes,
        ply_report,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"AERIAL_DIAGNOSTIC_{result['quality_status']} "
        f"close_ssim={result['metrics']['close_median_ssim']:.6f} "
        f"close_psnr={result['metrics']['close_median_psnr']:.6f} "
        f"control_ssim={result['metrics']['control_median_ssim']:.6f}"
    )


if __name__ == "__main__":
    main()
