"""Keep KIRI's Gaussian camera sockets synchronized with the active render camera."""

import bpy
from bpy.app.handlers import persistent


def _set_input(modifier, socket_id, value):
    inputs = getattr(getattr(modifier, "properties", None), "inputs", None)
    if inputs is not None:
        inputs[socket_id]["value"] = value
    else:
        modifier[socket_id] = value


def _write_matrix(modifier, first_socket, matrix):
    socket = first_socket
    for column in range(4):
        for row in range(4):
            _set_input(modifier, f"Socket_{socket}", float(matrix[row][column]))
            socket += 1


def splatitup_kiri_sync(scene, *_args):
    camera = scene.camera
    splat = bpy.data.objects.get("SPLATITUP_GAUSSIAN")
    if camera is None or splat is None:
        return
    modifier = splat.modifiers.get("KIRI_3DGS_Render_GN")
    if modifier is None:
        return

    width = max(1, int(scene.render.resolution_x * scene.render.resolution_percentage / 100))
    height = max(1, int(scene.render.resolution_y * scene.render.resolution_percentage / 100))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    # KIRI's node graph is evaluated under the splat object's transform, so the
    # render-camera world-to-view matrix must not multiply that transform twice.
    view_matrix = camera.matrix_world.inverted()
    projection_matrix = camera.calc_matrix_camera(
        depsgraph,
        x=width,
        y=height,
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y,
    )
    _set_input(modifier, "Socket_54", False)
    _write_matrix(modifier, 2, view_matrix)
    _write_matrix(modifier, 18, projection_matrix)
    _set_input(modifier, "Socket_34", width)
    _set_input(modifier, "Socket_35", height)
    splat.update_tag(refresh={"DATA"})


@persistent
def _frame_change(scene, *_args):
    splatitup_kiri_sync(scene)


@persistent
def _render_pre(scene, *_args):
    splatitup_kiri_sync(scene)


def register():
    for handlers, function in (
        (bpy.app.handlers.frame_change_post, _frame_change),
        (bpy.app.handlers.render_pre, _render_pre),
    ):
        for existing in list(handlers):
            if getattr(existing, "__name__", "") == function.__name__:
                handlers.remove(existing)
        handlers.append(function)
    splatitup_kiri_sync(bpy.context.scene)


register()
