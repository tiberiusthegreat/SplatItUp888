#!/usr/bin/env python3
"""Fully validate a standard binary SH3 Gaussian PLY payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


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
EXPECTED_REST_PROPERTIES = {f"f_rest_{index}" for index in range(45)}

PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path) -> tuple[str, int]:
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
                    return bytes(data[: end + 1]).decode("ascii", errors="strict"), end + 1
    raise ValueError("PLY header is missing or larger than 128 KiB")


def vertex_layout(header: str) -> tuple[int, list[tuple[str, str]]]:
    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    current_element: str | None = None
    for line in header.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                if vertex_count is not None:
                    raise ValueError("PLY declares more than one vertex element")
                vertex_count = int(parts[2])
        elif parts and parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3 or parts[1] == "list":
                raise ValueError("List properties are not supported in the Gaussian vertex element")
            if parts[1] not in PLY_TYPES:
                raise ValueError(f"Unsupported PLY scalar type: {parts[1]}")
            properties.append((parts[2], PLY_TYPES[parts[1]]))
    if vertex_count is None:
        raise ValueError("PLY does not declare a vertex element")
    return vertex_count, properties


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    if not args.ply.is_file():
        raise SystemExit(f"PLY does not exist: {args.ply}")

    header, payload_offset = read_header(args.ply)
    format_lines = [line for line in header.splitlines() if line.startswith("format ")]
    if format_lines != ["format binary_little_endian 1.0"]:
        raise SystemExit(f"Expected binary_little_endian PLY; found {format_lines}")

    vertex_count, layout = vertex_layout(header)
    property_names = [name for name, _data_type in layout]
    property_set = set(property_names)
    missing = sorted(REQUIRED_PROPERTIES - property_set)
    duplicate_properties = sorted({name for name in property_names if property_names.count(name) > 1})
    rest_property_set = {name for name in property_names if name.startswith("f_rest_")}
    missing_rest_properties = sorted(EXPECTED_REST_PROPERTIES - rest_property_set)
    unexpected_rest_properties = sorted(rest_property_set - EXPECTED_REST_PROPERTIES)
    gaussian_properties = REQUIRED_PROPERTIES | EXPECTED_REST_PROPERTIES
    layout_types = {name: data_type for name, data_type in layout}
    invalid_gaussian_property_types = sorted(
        name for name in gaussian_properties if name in layout_types and layout_types[name] != "<f4"
    )
    dtype = np.dtype(layout)
    expected_payload_bytes = vertex_count * dtype.itemsize
    available_payload_bytes = args.ply.stat().st_size - payload_offset
    payload_complete = available_payload_bytes == expected_payload_bytes

    nonfinite_vertices = 0
    zero_quaternions = 0
    bounds = None
    if payload_complete and vertex_count:
        vertices = np.memmap(
            args.ply,
            dtype=dtype,
            mode="r",
            offset=payload_offset,
            shape=(vertex_count,),
        )
        invalid = np.zeros(vertex_count, dtype=bool)
        for name in property_names:
            if np.issubdtype(vertices.dtype[name], np.floating):
                invalid |= ~np.isfinite(vertices[name])
        nonfinite_vertices = int(np.count_nonzero(invalid))
        if not missing:
            quaternion_norms = np.sqrt(
                vertices["rot_0"].astype(np.float64) ** 2
                + vertices["rot_1"].astype(np.float64) ** 2
                + vertices["rot_2"].astype(np.float64) ** 2
                + vertices["rot_3"].astype(np.float64) ** 2
            )
            zero_quaternions = int(np.count_nonzero(quaternion_norms <= 1e-12))
            bounds = {
                axis: {
                    "minimum": float(np.min(vertices[axis])),
                    "maximum": float(np.max(vertices[axis])),
                }
                for axis in ("x", "y", "z")
            }

    valid = (
        vertex_count >= 10_000
        and not missing
        and not duplicate_properties
        and not missing_rest_properties
        and not unexpected_rest_properties
        and not invalid_gaussian_property_types
        and payload_complete
        and nonfinite_vertices == 0
        and zero_quaternions == 0
    )
    report = {
        "path": str(args.ply.resolve()),
        "bytes": args.ply.stat().st_size,
        "sha256": sha256_file(args.ply),
        "format": "binary_little_endian 1.0",
        "header_bytes": payload_offset,
        "vertex_count": vertex_count,
        "vertex_stride_bytes": dtype.itemsize,
        "expected_vertex_payload_bytes": expected_payload_bytes,
        "available_payload_bytes": available_payload_bytes,
        "payload_complete": payload_complete,
        "properties": property_names,
        "missing_required_properties": missing,
        "duplicate_properties": duplicate_properties,
        "missing_spherical_harmonic_properties": missing_rest_properties,
        "unexpected_spherical_harmonic_properties": unexpected_rest_properties,
        "invalid_gaussian_property_types": invalid_gaussian_property_types,
        "spherical_harmonic_rest_properties": len(rest_property_set),
        "spherical_harmonic_bands": 3 if rest_property_set == EXPECTED_REST_PROPERTIES else None,
        "nonfinite_vertices": nonfinite_vertices,
        "zero_length_quaternions": zero_quaternions,
        "position_bounds": bounds,
        "valid_gaussian_ply": valid,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not valid:
        raise SystemExit(f"Invalid Gaussian PLY: {report}")
    print(
        f"Valid full SH3 Gaussian PLY: {vertex_count:,} splats, "
        f"{report['bytes']:,} bytes, sha256 {report['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
