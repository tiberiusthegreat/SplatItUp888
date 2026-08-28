#!/usr/bin/env python3
"""Build a self-contained KIRI 3DGS Blender scene with the solved camera path."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


REQUIRED_PROPERTIES = {
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

CAMERA_MODELS = {
    "SIMPLE_PINHOLE": (3, (0, 0, 1, 2)),
    "SIMPLE_RADIAL": (4, (0, 0, 1, 2)),
    "RADIAL": (5, (0, 0, 1, 2)),
}


def parse_args() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--model-text", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--ply-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--sync-script", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("Object", "Walkthrough", "House", "AerialExterior"),
        required=True,
    )
    return parser.parse_args(arguments)


def read_cameras(path: Path) -> dict[int, dict[str, object]]:
    cameras: dict[int, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(value) for value in parts[4:]]
        if model not in CAMERA_MODELS:
            raise RuntimeError(f"Unsupported COLMAP camera model for Blender: {model}")
        expected, indices = CAMERA_MODELS[model]
        if len(params) != expected:
            raise RuntimeError(f"COLMAP camera {camera_id} has {len(params)} parameters; expected {expected}")
        fx_index, fy_index, cx_index, cy_index = indices
        cameras[camera_id] = {
            "camera_id": camera_id,
            "model": model,
            "width": width,
            "height": height,
            "fx": params[fx_index],
            "fy": params[fy_index],
            "cx": params[cx_index],
            "cy": params[cy_index],
        }
    return cameras


def read_images(path: Path) -> list[dict[str, object]]:
    images: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        lines = iter(handle)
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=9)
            if len(parts) < 10:
                raise RuntimeError(f"Malformed COLMAP image line: {line}")
            images.append(
                {
                    "image_id": int(parts[0]),
                    "qvec": tuple(float(value) for value in parts[1:5]),
                    "tvec": tuple(float(value) for value in parts[5:8]),
                    "camera_id": int(parts[8]),
                    "name": parts[9],
                }
            )
            next(lines, None)  # POINTS2D observations, not needed for the handoff.
    return sorted(images, key=lambda image: str(image["name"]).lower())


def colmap_camera_matrix(qvec, tvec) -> Matrix:
    rotation_world_to_camera = Quaternion(qvec).to_matrix()
    rotation_camera_to_world = rotation_world_to_camera.transposed()
    center = -(rotation_camera_to_world @ Vector(tvec))
    camera_to_world = rotation_camera_to_world.to_4x4()
    camera_to_world.translation = center
    # OpenCV camera: +X right, +Y down, +Z forward.
    # Blender camera: +X right, +Y up, -Z forward.
    camera_axes = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
    return camera_to_world @ camera_axes


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.name != "Collection":
            bpy.data.collections.remove(collection)
    default = bpy.data.collections.get("Collection")
    if default:
        default.name = "00_GAUSSIAN_SPLAT"


def create_collection(name: str):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def move_to_collection(obj, collection) -> None:
    for existing in list(obj.users_collection):
        existing.objects.unlink(obj)
    collection.objects.link(obj)


def import_splat(path: Path, collection):
    if not hasattr(bpy.ops.sna, "dgs_render_import_ply_e0a3a"):
        raise RuntimeError("The installed KIRI 3DGS Render add-on is not active")
    props = bpy.context.scene.sna_dgs_scene_properties
    props.import_face_vert = "Verts"
    props.import_uv = False
    props.import_proxy = False
    result = bpy.ops.sna.dgs_render_import_ply_e0a3a(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError(f"KIRI PLY import failed: {result}")
    splat = bpy.context.active_object
    if splat is None or splat.type != "MESH":
        raise RuntimeError("KIRI did not leave an active Gaussian mesh")
    splat.name = "SPLATITUP_GAUSSIAN"
    splat.data.name = "SPLATITUP_GAUSSIAN_DATA"
    splat["source_ply"] = str(path.resolve())
    splat["gaussian_count"] = len(splat.data.vertices)
    splat["update_rot_to_cam"] = True
    render_modifier = splat.modifiers.get("KIRI_3DGS_Render_GN")
    if render_modifier is None:
        raise RuntimeError("KIRI render modifier is missing after import")
    render_modifier["Socket_50"] = 0  # Enable camera updates.
    render_modifier.show_viewport = True
    render_modifier.show_render = True
    move_to_collection(splat, collection)
    return splat


def configure_camera(camera_data, calibration: dict[str, object]) -> None:
    width = float(calibration["width"])
    height = float(calibration["height"])
    sensor_width = 36.0
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = sensor_width
    camera_data.lens = float(calibration["fx"]) * sensor_width / width
    camera_data.shift_x = (width * 0.5 - float(calibration["cx"])) / width
    camera_data.shift_y = (float(calibration["cy"]) - height * 0.5) / width


def action_fcurves(action):
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    curves = []
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for channelbag in getattr(strip, "channelbags", []):
                curves.extend(channelbag.fcurves)
    return curves


def create_camera_path(images, cameras, world, collection):
    camera_data = bpy.data.cameras.new("COLMAP_CAPTURE_CAMERA_DATA")
    camera_data.display_size = 0.25
    camera = bpy.data.objects.new("COLMAP_CAPTURE_CAMERA", camera_data)
    collection.objects.link(camera)
    camera.parent = world
    camera.rotation_mode = "QUATERNION"

    centers: list[Vector] = []
    index_records = []
    for frame, image in enumerate(images, start=1):
        calibration = cameras[int(image["camera_id"])]
        matrix = colmap_camera_matrix(image["qvec"], image["tvec"])
        camera.matrix_basis = matrix
        configure_camera(camera_data, calibration)
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        camera_data.keyframe_insert(data_path="lens", frame=frame)
        camera_data.keyframe_insert(data_path="shift_x", frame=frame)
        camera_data.keyframe_insert(data_path="shift_y", frame=frame)
        centers.append(matrix.translation.copy())
        index_records.append({"frame": frame, "image": image["name"], "image_id": image["image_id"]})

    for animated in (camera, camera_data):
        animation = animated.animation_data
        action = animation.action if animation else None
        if action:
            for curve in action_fcurves(action):
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"

    curve_data = bpy.data.curves.new("COLMAP_CAPTURE_PATH_DATA", type="CURVE")
    curve_data.dimensions = "3D"
    path_spline = curve_data.splines.new("POLY")
    path_spline.points.add(len(centers) - 1)
    for point, center in zip(path_spline.points, centers):
        point.co = (*center, 1.0)
    path = bpy.data.objects.new("COLMAP_CAPTURE_PATH", curve_data)
    collection.objects.link(path)
    path.parent = world
    path.hide_render = True
    curve_data.bevel_depth = 0.004
    curve_data.bevel_resolution = 1

    text = bpy.data.texts.new("COLMAP_CAMERA_INDEX.json")
    text.write(json.dumps(index_records, indent=2))
    return camera, path, centers


def embed_sync_script(path: Path) -> dict[str, object]:
    source = path.read_text(encoding="utf-8")
    text = bpy.data.texts.get("SPLATITUP_CAMERA_SYNC.py") or bpy.data.texts.new("SPLATITUP_CAMERA_SYNC.py")
    text.clear()
    text.write(source)
    text.use_module = True
    namespace: dict[str, object] = {"__name__": "SPLATITUP_CAMERA_SYNC"}
    exec(compile(source, "SPLATITUP_CAMERA_SYNC.py", "exec"), namespace)
    return namespace


def render_proof(scene, destination: Path, sync_namespace: dict[str, object]) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = (
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.render.resolution_percentage,
        scene.render.filepath,
        scene.render.image_settings.file_format,
    )
    aspect = original[1] / max(1, original[0])
    scene.render.resolution_x = 640
    scene.render.resolution_y = max(180, round(640 * aspect))
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(destination)
    scene.frame_set(1)
    sync_namespace["splatitup_kiri_sync"](scene)
    bpy.ops.render.render(write_still=True)

    render_result = bpy.data.images.load(str(destination), check_existing=False)
    try:
        _first_pixel = render_result.pixels[0]  # Force lazy disk-backed image loading.
    except (IndexError, RuntimeError) as error:
        raise RuntimeError("Blender could not reload the proof render for validation") from error
    pixels = np.empty(len(render_result.pixels), dtype=np.float32)
    render_result.pixels.foreach_get(pixels)
    rgb = pixels.reshape(-1, 4)[:, :3]
    luminance = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    proof = {
        "path": str(destination.resolve()),
        "width": scene.render.resolution_x,
        "height": scene.render.resolution_y,
        "luminance_standard_deviation": round(float(np.std(luminance)), 8),
        "luminance_range": round(float(np.max(luminance) - np.min(luminance)), 8),
    }
    proof["nonblank"] = (
        proof["luminance_standard_deviation"] > 0.01
        and proof["luminance_range"] > 0.10
    )
    if not proof["nonblank"]:
        raise RuntimeError(f"KIRI proof render is effectively blank: {proof}")
    bpy.data.images.remove(render_result)

    (
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.render.resolution_percentage,
        scene.render.filepath,
        scene.render.image_settings.file_format,
    ) = original
    sync_namespace["splatitup_kiri_sync"](scene)
    return proof


def create_readme(profile: str, image_count: int, excluded_count: int) -> None:
    text = bpy.data.texts.new("SPLATITUP_README.txt")
    text.write(
        "SplatItUp888 Blender handoff\n\n"
        f"Profile: {profile}\n"
        f"Solved camera poses: {image_count}\n\n"
        f"Rejected isolated camera-pose outliers: {excluded_count}\n\n"
        "COLMAP_CAPTURE_CAMERA follows the original solved capture path.\n"
        "Duplicate it or create a new camera for cinematic moves.\n"
        "SPLATITUP_WORLD parents both splat and camera; rotate or scale that Empty only.\n"
        "Changing only the splat transform will break camera alignment.\n"
        "The camera-sync script updates KIRI on frame changes and before rendering.\n"
        "If Blender blocks embedded scripts, mark this file trusted and reload it.\n"
    )


def main() -> None:
    args = parse_args()
    for path in (
        args.ply,
        args.model_text / "cameras.txt",
        args.model_text / "images.txt",
        args.pose_report,
        args.ply_report,
        args.sync_script,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    cameras = read_cameras(args.model_text / "cameras.txt")
    images = read_images(args.model_text / "images.txt")
    pose_report = json.loads(args.pose_report.read_text(encoding="utf-8-sig"))
    ply_report = json.loads(args.ply_report.read_text(encoding="utf-8-sig"))
    if not ply_report.get("valid_gaussian_ply"):
        raise RuntimeError("The Gaussian PLY report has not passed integrity validation")
    if not pose_report.get("quality_gates", {}).get("overall_pass"):
        raise RuntimeError("The reconstruction report has not passed all pose gates")
    excluded_names = set(
        pose_report.get("quality_gates", {})
        .get("trajectory_continuity", {})
        .get("excluded_isolated_poses", [])
    )
    images = [image for image in images if str(image["name"]) not in excluded_names]
    if len(images) < 2:
        raise RuntimeError("At least two registered COLMAP cameras are required")

    clear_scene()
    splat_collection = bpy.data.collections.get("00_GAUSSIAN_SPLAT")
    camera_collection = create_collection("10_SOLVED_CAMERA_PATH")
    world = bpy.data.objects.new("SPLATITUP_WORLD", None)
    bpy.context.scene.collection.objects.link(world)
    world.empty_display_type = "ARROWS"
    world.empty_display_size = 0.5
    world.matrix_world = Matrix.Rotation(math.radians(90.0), 4, "X")
    world["instruction"] = "Rotate or scale this parent to keep the splat and solved camera aligned."
    world["coordinate_conversion"] = "Source Y-up to Blender Z-up"

    splat = import_splat(args.ply, splat_collection)
    splat.parent = world
    splat.matrix_parent_inverse = Matrix.Identity(4)
    splat.matrix_basis = Matrix.Identity(4)
    camera, path, centers = create_camera_path(images, cameras, world, camera_collection)
    create_readme(args.profile, len(images), len(excluded_names))

    scene = bpy.context.scene
    scene.camera = camera
    scene.frame_start = 1
    scene.frame_end = len(images)
    scene.render.fps = 24
    first_calibration = cameras[int(images[0]["camera_id"])]
    scene.render.resolution_x = int(first_calibration["width"])
    scene.render.resolution_y = int(first_calibration["height"])
    scene.render.resolution_percentage = 50
    scene.frame_set(1)
    for name, frame in (("CAPTURE_START", 1), ("CAPTURE_MIDDLE", max(1, len(images) // 2)), ("CAPTURE_END", len(images))):
        marker = scene.timeline_markers.new(name, frame=frame)
        marker.camera = camera

    sync_namespace = embed_sync_script(args.sync_script)
    bpy.ops.object.select_all(action="DESELECT")
    camera.select_set(True)
    bpy.context.view_layer.objects.active = camera

    attributes = set(splat.data.attributes.keys())
    rest_properties = sorted(name for name in attributes if name.startswith("f_rest_"))
    expected_modifiers = {
        "KIRI_3DGS_Render_GN",
        "KIRI_3DGS_Sorter_GN",
        "KIRI_3DGS_Adjust_Colour_And_Material",
        "KIRI_3DGS_Write F_DC_And_Merge",
    }
    missing_properties = sorted(REQUIRED_PROPERTIES - attributes)
    missing_modifiers = sorted(expected_modifiers - set(splat.modifiers.keys()))
    finite_centers = all(all(math.isfinite(value) for value in center) for center in centers)
    mechanical_pass = (
        len(splat.data.vertices) == int(ply_report["vertex_count"])
        and not missing_properties
        and len(rest_properties) == 45
        and not missing_modifiers
        and finite_centers
        and splat.matrix_basis == Matrix.Identity(4)
    )
    if not mechanical_pass:
        raise RuntimeError(
            f"Blender validation failed: missing properties={missing_properties}, "
            f"SH={len(rest_properties)}, missing modifiers={missing_modifiers}, finite cameras={finite_centers}"
        )

    render_proof_report = render_proof(scene, args.preview, sync_namespace)
    for image in bpy.data.images:
        if image.source == "FILE" and not image.packed_file:
            image.pack()
    unpacked_images = [
        image.filepath for image in bpy.data.images if image.source == "FILE" and not image.packed_file
    ]
    if unpacked_images:
        raise RuntimeError(f"Blender handoff still has unpacked external images: {unpacked_images}")
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output), check_existing=False)
    report = {
        "status": "MECHANICAL PASS",
        "blender_version": bpy.app.version_string,
        "profile": args.profile,
        "source_ply": str(args.ply.resolve()),
        "output_blend": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "gaussian_count": len(splat.data.vertices),
        "spherical_harmonic_rest_properties": len(rest_properties),
        "kiri_modifiers": sorted(expected_modifiers),
        "solved_camera_poses": len(images),
        "excluded_isolated_camera_poses": sorted(excluded_names),
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "camera_transforms_finite": finite_centers,
        "splat_local_transform_identity": True,
        "external_images_packed": True,
        "shared_world_coordinate_conversion": "Source Y-up to Blender Z-up",
        "render_proof": render_proof_report,
        "visual_approval": "AWAITING USER APPROVAL",
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SPLATITUP_BLENDER_REPORT=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
