from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "pipeline" / "build_aerial_qc_sheets.py"
SPEC = importlib.util.spec_from_file_location("build_aerial_qc_sheets", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AerialQcSheetTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path]:
        references = root / "references"
        renders = root / "renders"
        references.mkdir()
        renders.mkdir()
        records = [
            ("frame_close_a", "#ef4444", "#7f1d1d", 0.41, 19.5),
            ("frame_close_b", "#22c55e", "#14532d", 0.52, 22.0),
            ("frame_control", "#3b82f6", "#1e3a8a", 0.71, 28.0),
        ]
        for stem, source_color, render_color, _ssim, _psnr in records:
            Image.new("RGB", (640, 360), source_color).save(references / f"{stem}.jpg")
            Image.new("RGB", (640, 360), render_color).save(renders / f"{stem}.png")
        gate = {
            "quality_status": "MECHANICAL_FAIL",
            "close_images": [
                {"image": f"{stem}.png", "reference": f"{stem}.jpg", "ssim": ssim, "psnr": psnr}
                for stem, _source, _render, ssim, psnr in records[:2]
            ],
            "control_images": [
                {"image": f"{stem}.png", "reference": f"{stem}.jpg", "ssim": ssim, "psnr": psnr}
                for stem, _source, _render, ssim, psnr in records[2:]
            ],
        }
        gate_path = root / "gate.json"
        gate_path.write_text(json.dumps(gate), encoding="utf-8")
        return gate_path, references, renders

    def run_cli(self, gate: Path, references: Path, renders: Path, output: Path):
        return subprocess.run(
            [
                sys.executable, str(SCRIPT), "--gate", str(gate),
                "--reference-images", str(references), "--renders", str(renders),
                "--output", str(output),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_writes_ordered_individual_close_and_master_sheets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate, references, renders = self.fixture(root)
            output = root / "review"
            result = self.run_cli(gate, references, renders, output)
            self.assertEqual(0, result.returncode, result.stderr)
            rows = sorted((output / "individual").glob("*.jpg"))
            self.assertEqual(
                [
                    "01-close-frame_close_a-1920w.jpg",
                    "02-close-frame_close_b-1920w.jpg",
                    "03-control-frame_control-1920w.jpg",
                ],
                [path.name for path in rows],
            )
            for path in rows:
                with Image.open(path) as image:
                    self.assertEqual((MODULE.WIDTH, MODULE.ROW_HEIGHT), image.size)
            with Image.open(output / "aerial-qc-close-only-1920w.jpg") as close:
                self.assertEqual(
                    (MODULE.WIDTH, MODULE.SHEET_HEADER + 2 * MODULE.ROW_HEIGHT), close.size
                )
            with Image.open(output / "aerial-qc-master-1920w.jpg") as master:
                self.assertEqual(
                    (MODULE.WIDTH, MODULE.SHEET_HEADER + 3 * MODULE.ROW_HEIGHT), master.size
                )

    def test_accepts_current_brush_double_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate, references, renders = self.fixture(root)
            payload = json.loads(gate.read_text(encoding="utf-8"))
            for record in payload["close_images"] + payload["control_images"]:
                old_path = renders / str(record["image"])
                new_name = f"{record['reference']}.png"
                old_path.rename(renders / new_name)
                record["image"] = new_name
            gate.write_text(json.dumps(payload), encoding="utf-8")

            output = root / "review"
            result = self.run_cli(gate, references, renders, output)

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(3, len(list((output / "individual").glob("*.jpg"))))

    def test_row_labels_include_status_and_metrics(self) -> None:
        record = {
            "category": "CLOSE", "image": "frame.png", "reference": "frame.jpg",
            "ssim": 0.4567, "psnr": 21.234,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (320, 180), "red").save(root / "frame.jpg")
            Image.new("RGB", (320, 180), "blue").save(root / "frame.png")
            row = MODULE.comparison_row(record, "MECHANICAL_FAIL", 1, 1, 1, root, root)
            self.assertEqual((MODULE.WIDTH, MODULE.ROW_HEIGHT), row.size)
            # Text is rendered into the header; it must not remain the flat background.
            self.assertGreater(len(row.crop((0, 0, MODULE.WIDTH, MODULE.ROW_HEADER)).getcolors(maxcolors=10000)), 2)

    def test_duplicate_stems_fail_before_output_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate, references, renders = self.fixture(root)
            payload = json.loads(gate.read_text(encoding="utf-8"))
            payload["control_images"][0] = dict(payload["close_images"][0])
            gate.write_text(json.dumps(payload), encoding="utf-8")
            output = root / "review"
            result = self.run_cli(gate, references, renders, output)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("duplicate image stems", result.stderr)
            self.assertFalse(output.exists())

    def test_missing_image_fails_before_output_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate, references, renders = self.fixture(root)
            (renders / "frame_close_b.png").unlink()
            output = root / "review"
            result = self.run_cli(gate, references, renders, output)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("Missing QC input image", result.stderr)
            self.assertFalse(output.exists())

    def test_malformed_gate_fails_before_output_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate, references, renders = self.fixture(root)
            gate.write_text('{"quality_status": ""}', encoding="utf-8")
            output = root / "review"
            result = self.run_cli(gate, references, renders, output)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("quality_status", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
