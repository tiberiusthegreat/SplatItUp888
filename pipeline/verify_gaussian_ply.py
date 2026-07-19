#!/usr/bin/env python3
"""Verify that a PLY contains the attributes expected by 3DGS editors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_PROPERTIES = {
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
}


def read_header(path: Path) -> str:
    data = bytearray()
    with path.open("rb") as handle:
        while len(data) < 131072:
            chunk = handle.read(4096)
            if not chunk:
                break
            data.extend(chunk)
            marker = data.find(b"end_header")
            if marker >= 0:
                end = data.find(b"\n", marker)
                if end >= 0:
                    return bytes(data[: end + 1]).decode("ascii", errors="strict")
    raise ValueError("PLY header is missing or larger than 128 KiB")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    if not args.ply.is_file():
        raise SystemExit(f"PLY does not exist: {args.ply}")

    header = read_header(args.ply)
    lines = header.splitlines()
    vertex_lines = [line for line in lines if line.startswith("element vertex ")]
    if len(vertex_lines) != 1:
        raise SystemExit("PLY does not declare exactly one vertex element")
    vertex_count = int(vertex_lines[0].split()[-1])
    properties = {
        line.split()[-1]
        for line in lines
        if line.startswith("property ") and not line.startswith("property list ")
    }
    missing = sorted(REQUIRED_PROPERTIES - properties)
    report = {
        "path": str(args.ply.resolve()),
        "bytes": args.ply.stat().st_size,
        "vertex_count": vertex_count,
        "properties": sorted(properties),
        "missing_required_properties": missing,
        "valid_gaussian_ply": vertex_count >= 10000 and not missing,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not report["valid_gaussian_ply"]:
        raise SystemExit(f"Invalid Gaussian PLY: {report}")
    print(f"Valid Gaussian PLY: {vertex_count:,} splats, {report['bytes']:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
