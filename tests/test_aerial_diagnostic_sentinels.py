from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "pipeline" / "select_aerial_diagnostic_sentinels.py"
SPEC = importlib.util.spec_from_file_location(
    "select_aerial_diagnostic_sentinels", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def healthy_distances(count: int = 30) -> list[float]:
    low_count = int(count * 0.4)
    high_count = int(count * 0.4)
    middle_count = count - low_count - high_count
    ranked = (
        [1.0 + 0.1 * index for index in range(low_count)]
        + [4.0 + 0.1 * index for index in range(middle_count)]
        + [8.0 + 0.1 * index for index in range(high_count)]
    )
    multiplier = 7 if count % 7 else 11
    return [ranked[(index * multiplier) % count] for index in range(count)]


def write_model(
    root: Path,
    holdout_distances: list[float],
    *,
    eval_split_every: int = 2,
    image_ids: list[int] | None = None,
    short_holdouts: set[int] | None = None,
    reverse_records: bool = False,
) -> Path:
    model = root / "model"
    model.mkdir(parents=True)
    registered_count = (len(holdout_distances) - 1) * eval_split_every + 1
    if image_ids is None:
        image_ids = list(range(1, registered_count + 1))
    if len(image_ids) != registered_count or image_ids != sorted(image_ids):
        raise AssertionError("Fixture image IDs must be sorted and match the image count")
    short_holdouts = short_holdouts or set()

    camera_records = [
        "1 PINHOLE 1920 1080 1000 1000 960 540",
        "2 PINHOLE 1920 1080 1000 1000 960 540",
    ]
    point_records = [
        f"{point_id} 0 0 0 255 255 255 0.1" for point_id in range(1, 21)
    ]
    image_records: list[tuple[str, str]] = []
    holdout_index = 0
    for registered_position, image_id in enumerate(image_ids, start=1):
        is_holdout = (registered_position - 1) % eval_split_every == 0
        if is_holdout:
            distance = holdout_distances[holdout_index]
            observation_count = 19 if holdout_index + 1 in short_holdouts else 20
            holdout_index += 1
        else:
            distance = 3.0
            observation_count = 20
        camera_id = 1 + (registered_position % 2)
        name = f"frame_{registered_position:06d}.jpg"
        header = (
            f"{image_id} 1 0 0 0 {-distance:.9f} 0 0 {camera_id} {name}"
        )
        observations = " ".join(
            f"{point_id}.0 {point_id}.0 {point_id}"
            for point_id in range(1, observation_count + 1)
        )
        image_records.append((header, observations))

    if reverse_records:
        camera_records.reverse()
        point_records.reverse()
        image_records.reverse()
    (model / "cameras.txt").write_text(
        "# cameras\n" + "\n".join(camera_records) + "\n", encoding="utf-8"
    )
    (model / "points3D.txt").write_text(
        "# points\n" + "\n".join(point_records) + "\n", encoding="utf-8"
    )
    image_lines = ["# registered images"]
    for header, observations in image_records:
        image_lines.extend((header, observations))
    (model / "images.txt").write_text(
        "\n".join(image_lines) + "\n", encoding="utf-8"
    )
    return model


class AerialDiagnosticSentinelTests(unittest.TestCase):
    def test_selects_exact_disjoint_sets_from_every_n_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = write_model(Path(directory), healthy_distances())
            report = MODULE.select_sentinels(model, 2)

        close = report["close_images"]
        controls = report["control_images"]
        pool = {item["name"] for item in report["prospective_holdouts"]}
        self.assertEqual(8, len(close))
        self.assertEqual(4, len(controls))
        self.assertEqual(12, len(set(close + controls)))
        self.assertTrue(set(close + controls) <= pool)
        self.assertTrue(
            all(
                (item["registered_position"] - 1) % 2 == 0
                for item in report["selected_close"] + report["selected_controls"]
            )
        )
        self.assertEqual(close, [item["name"] for item in report["selected_close"]])
        self.assertEqual(
            controls, [item["name"] for item in report["selected_controls"]]
        )
        self.assertEqual(8, report["counts"]["selected_close"])
        self.assertEqual(4, report["counts"]["selected_controls"])
        self.assertGreaterEqual(
            report["distance_separation"][
                "selected_control_to_close_median_ratio"
            ],
            1.25,
        )
        self.assertEqual(
            sorted(item["registered_position"] for item in report["selected_close"]),
            [item["registered_position"] for item in report["selected_close"]],
        )
        self.assertEqual(
            sorted(
                item["registered_position"] for item in report["selected_controls"]
            ),
            [item["registered_position"] for item in report["selected_controls"]],
        )

    def test_selection_is_stable_when_text_records_are_shuffled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_model(root / "first", healthy_distances())
            second = write_model(
                root / "second", healthy_distances(), reverse_records=True
            )
            first_report = MODULE.select_sentinels(first, 2)
            second_report = MODULE.select_sentinels(second, 2)

        self.assertEqual(first_report["close_images"], second_report["close_images"])
        self.assertEqual(
            first_report["control_images"], second_report["control_images"]
        )
        self.assertEqual(
            first_report["selected_close"], second_report["selected_close"]
        )
        self.assertEqual(
            first_report["selected_controls"], second_report["selected_controls"]
        )
        self.assertNotEqual(
            first_report["source_hashes"]["images.txt"],
            second_report["source_hashes"]["images.txt"],
        )

    def test_registered_positions_ignore_unregistered_image_id_gaps(self) -> None:
        distances = healthy_distances(20)
        registered_count = (len(distances) - 1) * 3 + 1
        image_ids = [2 + 3 * index for index in range(registered_count)]
        with tempfile.TemporaryDirectory() as directory:
            model = write_model(
                Path(directory),
                distances,
                eval_split_every=3,
                image_ids=image_ids,
                reverse_records=True,
            )
            report = MODULE.select_sentinels(model, 3)

        prospective = report["prospective_holdouts"]
        self.assertEqual(list(range(1, registered_count + 1, 3)), [
            item["registered_position"] for item in prospective
        ])
        self.assertEqual(image_ids[::3], [item["image_id"] for item in prospective])
        self.assertEqual(20, report["counts"]["prospective_holdouts"])

    def test_fails_when_too_few_holdouts_have_twenty_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = write_model(
                Path(directory), healthy_distances(20), short_holdouts={1}
            )
            with self.assertRaisesRegex(
                ValueError, "20 eligible prospective holdouts.*20 tracked observations"
            ):
                MODULE.select_sentinels(model, 2)

    def test_fails_when_selected_distance_ratio_is_ambiguous(self) -> None:
        distances = [10.0 + 0.1 * index for index in range(20)]
        with tempfile.TemporaryDirectory() as directory:
            model = write_model(Path(directory), distances)
            with self.assertRaisesRegex(ValueError, "ratio .* is below 1.250000"):
                MODULE.select_sentinels(model, 2)

    def test_fails_when_distance_bands_tie(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = write_model(Path(directory), [10.0] * 20)
            with self.assertRaisesRegex(ValueError, "strict separation"):
                MODULE.select_sentinels(model, 2)

    def test_cli_writes_hashes_positions_distances_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = write_model(root, healthy_distances())
            output = root / "sentinels.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--model-text",
                    str(model),
                    "--eval-split-every",
                    "2",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            expected_hash = hashlib.sha256(
                (model / "images.txt").read_bytes()
            ).hexdigest()

        self.assertIn("AERIAL_DIAGNOSTIC_SENTINELS_SELECTED", result.stdout)
        self.assertEqual(MODULE.ALGORITHM_VERSION, report["algorithm_version"])
        self.assertEqual(expected_hash, report["source_hashes"]["images.txt"])
        self.assertEqual(30, report["counts"]["prospective_holdouts"])
        for item in report["selected_close"] + report["selected_controls"]:
            self.assertIsInstance(item["registered_position"], int)
            self.assertIsInstance(
                item["median_camera_to_observed_point_distance"], float
            )


if __name__ == "__main__":
    unittest.main()
