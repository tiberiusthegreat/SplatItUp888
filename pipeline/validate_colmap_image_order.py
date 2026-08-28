#!/usr/bin/env python3
"""Fail if COLMAP image IDs do not follow the declared chronological order."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def read_expected(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_actual(database: Path) -> list[str]:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT name FROM images ORDER BY image_id"
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    args = parser.parse_args()

    expected = read_expected(args.image_list)
    actual = read_actual(args.database)
    if actual != expected:
        mismatch = next(
            (index for index, pair in enumerate(zip(actual, expected), start=1) if pair[0] != pair[1]),
            min(len(actual), len(expected)) + 1,
        )
        actual_name = actual[mismatch - 1] if mismatch <= len(actual) else "<missing>"
        expected_name = expected[mismatch - 1] if mismatch <= len(expected) else "<missing>"
        raise SystemExit(
            "COLMAP chronological image order mismatch at image_id "
            f"{mismatch}: expected {expected_name}, found {actual_name}; "
            f"expected {len(expected)} images, found {len(actual)}"
        )

    print(f"Verified chronological COLMAP image_id order for {len(actual)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
