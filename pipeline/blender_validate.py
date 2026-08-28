#!/usr/bin/env python3
"""Read-only validation of a generated SplatItUp888 scene in Blender 5.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy


EXPECTED_MODIFIERS = {
    "KIRI_3DGS_Render_GN",
    "KIRI_3DGS_Sorter_GN",
    "KIRI_3DGS_Adjust_Colour_And_Material",
    "KIRI_3DGS_Write F_DC_And_Merge",
}


def parse_args() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-report", type=Path, required=True)
    parser.add_argument("--sync-script", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(arguments)


def action_fcurves(action):
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    curves = []
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for channelbag in getattr(strip, "channelbags", []):
                curves.extend(channelbag.fcurves)
    return curves


def modifier_input(modifier, identifier: str):
    inputs = getattr(getattr(modifier, "properties", None), "inputs", None)
    if inputs is not None:
        return inputs[identifier]["value"]
    return modifier.get(identifier)


def normalized_source(source: str) -> str:
    return source.replace("\r\n", "\n").replace("\r", "\n")


def source_sha256(source: str) -> str:
    return hashlib.sha256(normalized_source(source).encode("utf-8")).hexdigest()


def matrix_values(matrix) -> tuple[float, ...]:
    return tuple(float(matrix[row][column]) for column in range(4) for row in range(4))


def modifier_matrix_values(modifier, first_socket: int) -> tuple[float, ...]:
    return tuple(float(modifier_input(modifier, f"Socket_{socket}")) for socket in range(first_socket, first_socket + 16))


def action_curve_counts(action) -> dict[str, int]:
    if action is None:
        return {}
    return {
        f"{curve.data_path}[{curve.array_index}]": len(curve.keyframe_points)
        for curve in action_fcurves(action)
    }


def main() -> None:
    args = parse_args()
    expected = json.loads(args.handoff_report.read_text(encoding="utf-8-sig"))
    splat = bpy.data.objects.get("SPLATITUP_GAUSSIAN")
    camera = bpy.data.objects.get("COLMAP_CAPTURE_CAMERA")
    world = bpy.data.objects.get("SPLATITUP_WORLD")
    path = bpy.data.objects.get("COLMAP_CAPTURE_PATH")
    if any(value is None for value in (splat, camera, world, path)):
        raise RuntimeError("Required SplatItUp888 Blender objects are missing")

    render_modifier = splat.modifiers.get("KIRI_3DGS_Render_GN")
    if render_modifier is None:
        raise RuntimeError("KIRI render modifier is missing")

    embedded_sync = bpy.data.texts.get("SPLATITUP_CAMERA_SYNC.py")
    if embedded_sync is None:
        raise RuntimeError("Embedded KIRI camera-sync script is missing")
    local_sync_source = args.sync_script.read_text(encoding="utf-8")
    embedded_sync_source = embedded_sync.as_string()
    embedded_sync_source_matches = source_sha256(embedded_sync_source) == source_sha256(local_sync_source)
    if not embedded_sync_source_matches:
        raise RuntimeError("Embedded KIRI camera-sync script does not match the validated local source")
    exec(
        compile(embedded_sync_source, "SPLATITUP_CAMERA_SYNC.py", "exec"),
        {"__name__": "__splatitup_validation__"},
    )

    scene = bpy.context.scene
    sample_frames = sorted({scene.frame_start, max(scene.frame_start, scene.frame_end // 2), scene.frame_end})
    view_errors = []
    projection_errors = []
    sampled_views = []
    for frame in sample_frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        expected_view = matrix_values(camera.matrix_world.inverted())
        expected_projection = matrix_values(
            camera.calc_matrix_camera(
                bpy.context.evaluated_depsgraph_get(),
                x=max(1, int(scene.render.resolution_x * scene.render.resolution_percentage / 100)),
                y=max(1, int(scene.render.resolution_y * scene.render.resolution_percentage / 100)),
                scale_x=scene.render.pixel_aspect_x,
                scale_y=scene.render.pixel_aspect_y,
            )
        )
        actual_view = modifier_matrix_values(render_modifier, 2)
        actual_projection = modifier_matrix_values(render_modifier, 18)
        view_errors.append(max(abs(actual - expected) for actual, expected in zip(actual_view, expected_view)))
        projection_errors.append(max(abs(actual - expected) for actual, expected in zip(actual_projection, expected_projection)))
        sampled_views.append(tuple(round(value, 7) for value in actual_view))

    expected_width = max(1, int(scene.render.resolution_x * scene.render.resolution_percentage / 100))
    expected_height = max(1, int(scene.render.resolution_y * scene.render.resolution_percentage / 100))
    camera_resolution_matches = (
        int(modifier_input(render_modifier, "Socket_34")) == expected_width
        and int(modifier_input(render_modifier, "Socket_35")) == expected_height
    )
    camera_action = camera.animation_data.action if camera.animation_data else None
    camera_data_action = camera.data.animation_data.action if camera.data.animation_data else None
    keyed_frames = set()
    if camera_action:
        for curve in action_fcurves(camera_action):
            keyed_frames.update(round(point.co.x) for point in curve.keyframe_points)
    rest_properties = [name for name in splat.data.attributes.keys() if name.startswith("f_rest_")]
    missing_modifiers = sorted(EXPECTED_MODIFIERS - set(splat.modifiers.keys()))
    expected_camera_keys = int(expected["solved_camera_poses"])
    transform_curve_counts = action_curve_counts(camera_action)
    intrinsic_curve_counts = action_curve_counts(camera_data_action)
    required_transform_curves = {
        *(f"location[{index}]" for index in range(3)),
        *(f"rotation_quaternion[{index}]" for index in range(4)),
    }
    required_intrinsic_curves = {"lens[0]", "shift_x[0]", "shift_y[0]"}
    camera_transform_curves_complete = all(
        transform_curve_counts.get(name) == expected_camera_keys for name in required_transform_curves
    )
    camera_intrinsic_curves_complete = all(
        intrinsic_curve_counts.get(name) == expected_camera_keys for name in required_intrinsic_curves
    )
    unpacked_external_images = [
        image.filepath for image in bpy.data.images if image.source == "FILE" and not image.packed_file
    ]
    report = {
        "status": "MECHANICAL PASS",
        "blender_version": bpy.app.version_string,
        "blend_file": str(Path(bpy.data.filepath).resolve()),
        "gaussian_count": len(splat.data.vertices),
        "spherical_harmonic_rest_properties": len(rest_properties),
        "missing_kiri_modifiers": missing_modifiers,
        "camera_parent": camera.parent.name if camera.parent else None,
        "splat_parent": splat.parent.name if splat.parent else None,
        "world_rotation_x_degrees": round(math.degrees(world.rotation_euler.x), 4),
        "camera_keyed_frames": len(keyed_frames),
        "camera_transform_curves_complete": camera_transform_curves_complete,
        "camera_intrinsic_curves_complete": camera_intrinsic_curves_complete,
        "timeline_end": bpy.context.scene.frame_end,
        "camera_update_mode": modifier_input(render_modifier, "Socket_50") if render_modifier else None,
        "embedded_sync_source_matches": embedded_sync_source_matches,
        "sync_sample_frames": sample_frames,
        "sync_distinct_view_matrices": len(set(sampled_views)),
        "sync_max_view_matrix_error": max(view_errors),
        "sync_max_projection_matrix_error": max(projection_errors),
        "sync_camera_resolution_matches": camera_resolution_matches,
        "external_libraries": [library.filepath for library in bpy.data.libraries],
        "unpacked_external_images": unpacked_external_images,
        "embedded_texts": sorted(bpy.data.texts.keys()),
        "render_proof_nonblank": bool(expected.get("render_proof", {}).get("nonblank")),
        "visual_approval": "AWAITING USER APPROVAL",
    }
    valid = (
        report["gaussian_count"] == int(expected["gaussian_count"])
        and report["spherical_harmonic_rest_properties"] == 45
        and not missing_modifiers
        and report["camera_parent"] == world.name
        and report["splat_parent"] == world.name
        and abs(report["world_rotation_x_degrees"] - 90.0) < 0.001
        and report["camera_keyed_frames"] == int(expected["solved_camera_poses"])
        and report["camera_transform_curves_complete"]
        and report["camera_intrinsic_curves_complete"]
        and report["timeline_end"] == int(expected["frame_end"])
        and report["camera_update_mode"] == 0
        and report["embedded_sync_source_matches"]
        and report["sync_distinct_view_matrices"] >= 2
        and report["sync_max_view_matrix_error"] <= 1e-5
        and report["sync_max_projection_matrix_error"] <= 1e-5
        and report["sync_camera_resolution_matches"]
        and not report["external_libraries"]
        and not report["unpacked_external_images"]
        and "SPLATITUP_CAMERA_SYNC.py" in report["embedded_texts"]
        and report["render_proof_nonblank"]
    )
    if not valid:
        report["status"] = "MECHANICAL FAIL"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SPLATITUP_BLENDER_OPEN_REPORT=" + json.dumps(report, sort_keys=True))
    if not valid:
        raise RuntimeError(f"Blender destination validation failed: {report}")


if __name__ == "__main__":
    main()
