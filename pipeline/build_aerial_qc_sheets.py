#!/usr/bin/env python3
"""Build iPad-readable source/render comparisons from an aerial gate report."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


WIDTH = 1920
PANEL_WIDTH = WIDTH // 2
PANEL_HEIGHT = 540
ROW_HEADER = 116
ROW_HEIGHT = ROW_HEADER + PANEL_HEIGHT
SHEET_HEADER = 180


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def simple_filename(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty filename")
    path = Path(value)
    if path.is_absolute() or path.name != value or value in {".", ".."}:
        raise ValueError(f"{label} must be a simple filename")
    return value


def image_stem(value: str) -> str:
    path = Path(value)
    stem = path.stem
    if path.suffix.casefold() == ".png" and Path(stem).suffix.casefold() in {".jpg", ".jpeg"}:
        return Path(stem).stem
    return stem


def validate_record(value: object, category: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Every {category} image record must be an object")
    image_name = simple_filename(value.get("image"), f"{category} image")
    reference_name = simple_filename(value.get("reference"), f"{category} reference")
    if image_stem(image_name).casefold() != image_stem(reference_name).casefold():
        raise ValueError(f"Render/reference stems do not match: {image_name}")
    metrics: dict[str, float] = {}
    for name in ("ssim", "psnr"):
        metric = value.get(name)
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError(f"Missing or invalid {name} for {image_name}")
        metric = float(metric)
        if not math.isfinite(metric):
            raise ValueError(f"Non-finite {name} for {image_name}")
        metrics[name] = metric
    if not 0.0 <= metrics["ssim"] <= 1.0 or not 0.0 <= metrics["psnr"] <= 100.0:
        raise ValueError(f"Metrics are outside their valid range for {image_name}")
    return {
        "category": category,
        "image": image_name,
        "reference": reference_name,
        **metrics,
    }


def load_gate(path: Path) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read gate JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Gate JSON must be an object")
    status = payload.get("quality_status")
    if not isinstance(status, str) or not status.strip():
        raise ValueError("Gate quality_status must be a non-empty string")
    groups: list[list[dict[str, object]]] = []
    for key, category in (("close_images", "CLOSE"), ("control_images", "CONTROL")):
        raw_records = payload.get(key)
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError(f"Gate {key} must be a non-empty list")
        groups.append([validate_record(record, category) for record in raw_records])
    close, controls = groups
    stems = [image_stem(str(record["image"])).casefold() for record in close + controls]
    if len(stems) != len(set(stems)):
        raise ValueError("Gate contains duplicate image stems")
    return status.strip(), close, controls


def validate_input_images(
    records: list[dict[str, object]], reference_dir: Path, render_dir: Path
) -> None:
    for record in records:
        for path in (
            reference_dir / str(record["reference"]),
            render_dir / str(record["image"]),
        ):
            if not path.is_file():
                raise ValueError(f"Missing QC input image: {path}")
            try:
                with Image.open(path) as source:
                    source.load()
            except OSError as exc:
                raise ValueError(f"Unreadable QC input image: {path}") from exc


def panel(path: Path, label: str) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), "#0b1018")
    result.paste(image, ((PANEL_WIDTH - image.width) // 2, (PANEL_HEIGHT - image.height) // 2))
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, PANEL_WIDTH, 40), fill="#000000")
    draw.text((18, 6), label, fill="#ffffff", font=font(24))
    return result


def comparison_row(
    record: dict[str, object], status: str, index: int, category_index: int,
    category_count: int, reference_dir: Path, render_dir: Path,
) -> Image.Image:
    result = Image.new("RGB", (WIDTH, ROW_HEIGHT), "#111827")
    draw = ImageDraw.Draw(result)
    stem = image_stem(str(record["image"]))
    draw.text(
        (28, 13),
        f"{record['category']} {category_index:02d}/{category_count:02d}  |  {stem}",
        fill="#ffffff",
        font=font(34),
    )
    draw.text(
        (28, 66),
        f"STATUS: {status}  |  SSIM {float(record['ssim']):.3f}  |  PSNR {float(record['psnr']):.2f} dB  |  QC #{index:02d}",
        fill="#fbbf24",
        font=font(27),
    )
    source_panel = panel(reference_dir / str(record["reference"]), "SOURCE REFERENCE")
    render_panel = panel(render_dir / str(record["image"]), "SPLAT RENDER")
    result.paste(source_panel, (0, ROW_HEADER))
    result.paste(render_panel, (PANEL_WIDTH, ROW_HEADER))
    return result


def save_jpeg(image: Image.Image, path: Path) -> None:
    image.save(path, format="JPEG", quality=94, subsampling=0, optimize=True)


def contact_sheet(rows: list[Path], title: str, status: str, destination: Path) -> None:
    result = Image.new("RGB", (WIDTH, SHEET_HEADER + ROW_HEIGHT * len(rows)), "#0f172a")
    draw = ImageDraw.Draw(result)
    draw.text((30, 28), title, fill="#ffffff", font=font(48))
    draw.text((30, 104), f"QUALITY STATUS: {status}", fill="#fbbf24", font=font(31))
    y = SHEET_HEADER
    for path in rows:
        with Image.open(path) as row:
            result.paste(row.convert("RGB"), (0, y))
        y += ROW_HEIGHT
    save_jpeg(result, destination)


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_") or "frame"


def build(gate: Path, reference_dir: Path, render_dir: Path, output_dir: Path) -> list[Path]:
    status, close, controls = load_gate(gate)
    records = close + controls
    validate_input_images(records, reference_dir, render_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(exist_ok=True)
    row_paths: list[Path] = []
    category_positions = {"CLOSE": 0, "CONTROL": 0}
    category_counts = {"CLOSE": len(close), "CONTROL": len(controls)}
    for index, record in enumerate(records, start=1):
        category = str(record["category"])
        category_positions[category] += 1
        row = comparison_row(
            record, status, index, category_positions[category], category_counts[category],
            reference_dir, render_dir,
        )
        path = individual_dir / f"{index:02d}-{category.lower()}-{safe_stem(image_stem(str(record['image'])))}-1920w.jpg"
        save_jpeg(row, path)
        row_paths.append(path)
    contact_sheet(
        row_paths[: len(close)], "AERIAL SPLAT QC - CLOSE VIEWS", status,
        output_dir / "aerial-qc-close-only-1920w.jpg",
    )
    contact_sheet(
        row_paths, "AERIAL SPLAT QC - CLOSE THEN CONTROLS", status,
        output_dir / "aerial-qc-master-1920w.jpg",
    )
    return row_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--reference-images", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.gate, args.reference_images, args.renders, args.output)
    print(f"AERIAL_QC_SHEETS_WRITTEN rows={len(rows)} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
