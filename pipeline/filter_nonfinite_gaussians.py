#!/usr/bin/env python3
"""Remove only verified transparent Brush tombstones from a Gaussian PLY."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


HEADER_LIMIT = 128 * 1024
CHUNK_RECORDS = 4096
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
TOMBSTONE_SCALE_PROPERTIES = ("scale_0", "scale_1", "scale_2")
TOMBSTONE_ALPHA_MAX = 1e-6
# Brush's observed float32 inverse-sigmoid marker for alpha 1e-6. Keep the
# serialized float32 value exact instead of recomputing it with wider
# intermediate precision.
TOMBSTONE_RAW_OPACITY_MAX = np.float32(-13.815508842468262)
TOMBSTONE_POLICY_ID = "brush-transparent-all-nan-scale-tombstone-v1"


class FilterError(ValueError):
    pass


def tombstone_mask(
    values: np.ndarray,
    properties: list[str],
    row_offset: int,
) -> np.ndarray:
    scale_indices = [properties.index(name) for name in TOMBSTONE_SCALE_PROPERTIES]
    non_scale_indices = [
        index for index, name in enumerate(properties) if name not in TOMBSTONE_SCALE_PROPERTIES
    ]
    opacity_index = properties.index("opacity")

    row_nonfinite = np.any(~np.isfinite(values), axis=1)
    if not np.any(row_nonfinite):
        return np.zeros(values.shape[0], dtype=bool)

    scale_values = values[:, scale_indices]
    scale_nan = np.isnan(scale_values)
    all_scales_nan = np.all(scale_nan, axis=1)
    non_scale_finite = np.all(np.isfinite(values[:, non_scale_indices]), axis=1)
    opacity_finite = np.isfinite(values[:, opacity_index])
    opacity_at_or_below_threshold = (
        opacity_finite & (values[:, opacity_index] <= TOMBSTONE_RAW_OPACITY_MAX)
    )
    tombstones = (
        row_nonfinite
        & all_scales_nan
        & non_scale_finite
        & opacity_at_or_below_threshold
    )

    unsafe = row_nonfinite & ~tombstones
    if np.any(unsafe):
        unsafe_indices = np.flatnonzero(unsafe)
        partial_scale_nan = (np.count_nonzero(scale_nan, axis=1) > 0) & ~all_scales_nan
        any_infinity = np.any(np.isinf(values), axis=1)
        non_scale_nonfinite = np.any(~np.isfinite(values[:, non_scale_indices]), axis=1)
        visible_opacity = opacity_finite & (
            values[:, opacity_index] > TOMBSTONE_RAW_OPACITY_MAX
        )
        raise FilterError(
            "Unsafe nonfinite Gaussian records found; refusing broad deletion: "
            f"count={unsafe_indices.size}, "
            f"first_vertex={row_offset + int(unsafe_indices[0])}, "
            f"not_all_scales_nan={int(np.count_nonzero(unsafe & ~all_scales_nan))}, "
            f"partial_scale_nan={int(np.count_nonzero(unsafe & partial_scale_nan))}, "
            f"any_infinity={int(np.count_nonzero(unsafe & any_infinity))}, "
            f"non_scale_nonfinite={int(np.count_nonzero(unsafe & non_scale_nonfinite))}, "
            f"nonfinite_opacity={int(np.count_nonzero(unsafe & ~opacity_finite))}, "
            f"visible_opacity={int(np.count_nonzero(unsafe & visible_opacity))}"
        )
    return tombstones


def read_header(path: Path) -> bytes:
    data = bytearray()
    with path.open("rb") as handle:
        while len(data) < HEADER_LIMIT:
            chunk = handle.read(4096)
            if not chunk:
                break
            data.extend(chunk)
            match = re.search(br"(?m)^end_header\r?\n", data)
            if match:
                return bytes(data[: match.end()])
    raise FilterError("PLY header is missing or larger than 128 KiB")


def parse_header(header: bytes) -> tuple[int, list[str], tuple[int, int]]:
    try:
        text = header.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise FilterError("PLY header is not ASCII") from exc

    lines = text.splitlines()
    if not lines or lines[0] != "ply":
        raise FilterError("PLY magic line is missing")
    if [line for line in lines if line.startswith("format ")] != [
        "format binary_little_endian 1.0"
    ]:
        raise FilterError("Expected exactly one binary_little_endian 1.0 format line")

    elements: list[tuple[str, int]] = []
    properties: list[str] = []
    current_element: str | None = None
    for line in lines:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element":
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise FilterError(f"Invalid element count: {parts[2]}") from exc
            if count < 0:
                raise FilterError("Element counts cannot be negative")
            current_element = parts[1]
            elements.append((current_element, count))
        elif parts and parts[0] == "property":
            if current_element != "vertex":
                raise FilterError("Properties outside the vertex element are not supported")
            if len(parts) != 3 or parts[1] == "list":
                raise FilterError("Only scalar vertex properties are supported")
            if parts[1] not in {"float", "float32"}:
                raise FilterError(f"Vertex property {parts[2]} is not float32")
            properties.append(parts[2])

    if len(elements) != 1 or elements[0][0] != "vertex":
        raise FilterError("Expected one vertex element and no other elements")
    vertex_count = elements[0][1]
    if vertex_count == 0:
        raise FilterError("Vertex element is empty")
    if len(properties) != len(set(properties)):
        raise FilterError("Duplicate vertex property names are not supported")

    property_set = set(properties)
    missing = sorted(REQUIRED_PROPERTIES - property_set)
    if missing:
        raise FilterError(f"Missing Gaussian properties: {missing}")
    rest_properties = {name for name in properties if name.startswith("f_rest_")}
    if rest_properties != EXPECTED_REST_PROPERTIES:
        missing_rest = sorted(EXPECTED_REST_PROPERTIES - rest_properties)
        unexpected_rest = sorted(rest_properties - EXPECTED_REST_PROPERTIES)
        raise FilterError(
            f"Invalid SH3 properties; missing={missing_rest}, unexpected={unexpected_rest}"
        )

    count_matches = list(
        re.finditer(br"(?m)^element[ \t]+vertex[ \t]+([0-9]+)[ \t]*\r?$", header)
    )
    if len(count_matches) != 1:
        raise FilterError("Could not uniquely locate the vertex count in the header")
    count_match = count_matches[0]
    if int(count_match.group(1)) != vertex_count:
        raise FilterError("Vertex count token does not match the parsed header")
    return vertex_count, properties, count_match.span(1)


def scan_payload(
    path: Path,
    header: bytes,
    vertex_count: int,
    properties: list[str],
) -> tuple[str, int, np.ndarray, float | None, float | None]:
    property_count = len(properties)
    stride = property_count * 4
    digest = hashlib.sha256(header)
    removed = 0
    distribution = np.zeros((property_count, 4), dtype=np.int64)
    removed_opacity_min: float | None = None
    removed_opacity_max: float | None = None
    opacity_index = properties.index("opacity")
    remaining = vertex_count
    row_offset = 0
    with path.open("rb") as handle:
        handle.seek(len(header))
        while remaining:
            rows = min(CHUNK_RECORDS, remaining)
            chunk = handle.read(rows * stride)
            if len(chunk) != rows * stride:
                raise FilterError("PLY payload ended before the declared vertex count")
            digest.update(chunk)
            values = np.frombuffer(chunk, dtype="<f4").reshape(rows, property_count)
            nonfinite = ~np.isfinite(values)
            tombstones = tombstone_mask(values, properties, row_offset)
            tombstone_count = int(np.count_nonzero(tombstones))
            removed += tombstone_count
            if tombstone_count:
                opacities = values[tombstones, opacity_index]
                chunk_min = float(np.min(opacities))
                chunk_max = float(np.max(opacities))
                removed_opacity_min = (
                    chunk_min
                    if removed_opacity_min is None
                    else min(removed_opacity_min, chunk_min)
                )
                removed_opacity_max = (
                    chunk_max
                    if removed_opacity_max is None
                    else max(removed_opacity_max, chunk_max)
                )
            distribution[:, 0] += np.count_nonzero(nonfinite, axis=0)
            distribution[:, 1] += np.count_nonzero(np.isnan(values), axis=0)
            distribution[:, 2] += np.count_nonzero(np.isposinf(values), axis=0)
            distribution[:, 3] += np.count_nonzero(np.isneginf(values), axis=0)
            remaining -= rows
            row_offset += rows
        if handle.read(1):
            raise FilterError("PLY contains trailing bytes after the vertex payload")
    return (
        digest.hexdigest(),
        removed,
        distribution,
        removed_opacity_min,
        removed_opacity_max,
    )


def write_filtered_payload(
    input_path: Path,
    output_path: Path,
    input_header: bytes,
    output_header: bytes,
    vertex_count: int,
    properties: list[str],
    expected_input_sha256: str,
    expected_retained: int,
) -> tuple[str, str, int]:
    property_count = len(properties)
    stride = property_count * 4
    input_digest = hashlib.sha256(input_header)
    output_digest = hashlib.sha256(output_header)
    retained_digest = hashlib.sha256()
    retained = 0
    remaining = vertex_count
    row_offset = 0

    with input_path.open("rb") as source, output_path.open("xb") as destination:
        source.seek(len(input_header))
        destination.write(output_header)
        while remaining:
            rows = min(CHUNK_RECORDS, remaining)
            chunk = source.read(rows * stride)
            if len(chunk) != rows * stride:
                raise FilterError("Input changed while the filtered PLY was being written")
            input_digest.update(chunk)
            values = np.frombuffer(chunk, dtype="<f4").reshape(rows, property_count)
            keep = ~tombstone_mask(values, properties, row_offset)
            kept = np.frombuffer(chunk, dtype=np.dtype((np.void, stride)), count=rows)[keep].tobytes()
            destination.write(kept)
            output_digest.update(kept)
            retained_digest.update(kept)
            retained += int(np.count_nonzero(keep))
            remaining -= rows
            row_offset += rows
        if source.read(1):
            raise FilterError("Input gained trailing bytes while the filtered PLY was being written")
        destination.flush()
        os.fsync(destination.fileno())

    if input_digest.hexdigest() != expected_input_sha256:
        raise FilterError("Input hash changed between validation and filtering")
    if retained != expected_retained:
        raise FilterError("Retained record count changed between validation and filtering")
    return output_digest.hexdigest(), retained_digest.hexdigest(), retained


def filter_ply(input_path: Path, output_path: Path, report_path: Path) -> dict[str, object]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    if not input_path.is_file():
        raise FilterError(f"Input PLY does not exist: {input_path}")
    if len({input_path, output_path, report_path}) != 3:
        raise FilterError("Input, output, and report paths must be different")
    if os.path.lexists(output_path):
        raise FilterError(f"Refusing to overwrite output: {output_path}")
    if os.path.lexists(report_path):
        raise FilterError(f"Refusing to overwrite report: {report_path}")

    header = read_header(input_path)
    vertex_count, properties, count_span = parse_header(header)
    stride = len(properties) * 4
    expected_size = len(header) + vertex_count * stride
    actual_size = input_path.stat().st_size
    if actual_size != expected_size:
        raise FilterError(
            f"Payload size mismatch: expected {expected_size} bytes, found {actual_size}"
        )

    (
        input_sha256,
        removed,
        distribution,
        removed_opacity_min,
        removed_opacity_max,
    ) = scan_payload(
        input_path, header, vertex_count, properties
    )
    retained = vertex_count - removed
    if retained == 0:
        raise FilterError("Filtering would remove every Gaussian")
    output_header = (
        header[: count_span[0]] + str(retained).encode("ascii") + header[count_span[1] :]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_sha256, retained_payload_sha256, written = write_filtered_payload(
            input_path,
            output_path,
            header,
            output_header,
            vertex_count,
            properties,
            input_sha256,
            retained,
        )
        output_bytes = output_path.stat().st_size
        expected_output_bytes = len(output_header) + retained * stride
        if output_bytes != expected_output_bytes:
            raise FilterError(
                f"Filtered payload size mismatch: expected {expected_output_bytes}, found {output_bytes}"
            )

        removed_distribution = {
            name: {
                "nonfinite_values": int(distribution[index, 0]),
                "nan_values": int(distribution[index, 1]),
                "positive_infinity_values": int(distribution[index, 2]),
                "negative_infinity_values": int(distribution[index, 3]),
            }
            for index, name in enumerate(properties)
            if distribution[index, 0]
        }
        report: dict[str, object] = {
            "schema_version": "splatitup-filter-nonfinite-gaussians-v2",
            "input": {
                "path": str(input_path),
                "bytes": actual_size,
                "sha256": input_sha256,
                "vertex_count": vertex_count,
                "header_bytes": len(header),
            },
            "output": {
                "path": str(output_path),
                "bytes": output_bytes,
                "sha256": output_sha256,
                "vertex_count": written,
                "header_bytes": len(output_header),
            },
            "vertex_stride_bytes": stride,
            "properties": properties,
            "removed_vertex_count": removed,
            "retained_vertex_count": retained,
            "removed_fraction": removed / vertex_count,
            "removed_nonfinite_property_distribution": removed_distribution,
            "removal_policy": {
                "policy_id": TOMBSTONE_POLICY_ID,
                "scale_properties": list(TOMBSTONE_SCALE_PROPERTIES),
                "required_scale_state": "all_nan",
                "non_scale_properties_must_be_finite": True,
                "infinity_allowed": False,
                "opacity_property": "opacity",
                "raw_opacity_max_inclusive": float(TOMBSTONE_RAW_OPACITY_MAX),
                "alpha_max": TOMBSTONE_ALPHA_MAX,
            },
            "removal_evidence": {
                "candidate_nonfinite_vertex_count": removed,
                "accepted_tombstone_vertex_count": removed,
                "rejected_nonfinite_vertex_count": 0,
                "all_scales_nan_vertex_count": removed,
                "finite_non_scale_vertex_count": removed,
                "opacity_at_or_below_threshold_vertex_count": removed,
                "infinity_value_count": int(
                    np.sum(distribution[:, 2]) + np.sum(distribution[:, 3])
                ),
                "removed_raw_opacity_min": removed_opacity_min,
                "removed_raw_opacity_max": removed_opacity_max,
            },
            "retained_payload_sha256": retained_payload_sha256,
            "retained_records_raw_copied": True,
            "retained_record_order_preserved": True,
            "header_change": "element vertex count only",
        }
        with report_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        return report
    except Exception:
        if os.path.lexists(output_path):
            output_path.unlink()
        if os.path.lexists(report_path):
            report_path.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = filter_ply(args.input, args.output, args.json)
    except (FilterError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"Filtered Gaussian PLY: {report['input']['vertex_count']:,} -> "
        f"{report['output']['vertex_count']:,} vertices; sha256 {report['output']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
