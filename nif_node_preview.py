"""Live viewport preview of NiSwitchNode and NiLODNode selections, and of NiBillboardNode rotation.

A timer reads the navigated 3D view's eye position, applies the engine's selection rule to every switch and
LOD Empty, and hides the subtrees of inactive children with `hide_set`. It only ever un-hides what it hid itself.
Explicit saves never keep a preview's hide state. Recovery state follows each changed object through undo and
duplication; loading a file written without save handlers restores its authored values.

Billboards turn through their delta rotation, which this preview owns while it runs: the author's own delta is
stashed and written back on restore, on save and before export. Editing a billboard's delta rotation while
previewing is overwritten.
"""

import math
import time
import traceback

import bpy
from bpy.app.handlers import persistent
from mathutils import Euler, Matrix, Quaternion, Vector

from . import nif_switch

INTERVAL = 0.1
FULL_REFRESH = 1.0  # also re-evaluate a still view, to pick up range edits and undo
BILLBOARD = "NiBillboardNode"

_views = {}  # (window index, area index) -> last seen view matrix
_focus = None  # key of the most recently navigated view
_active = {}  # switch or LOD name -> the shown child's name, or None
_last_full = 0.0
_stopping = False


def active_child_name(node):
    """The child the last preview tick showed for `node`, or None."""
    return _active.get(node.name)


def refresh():
    """Re-evaluate on the next tick even if no view moved, e.g. after a switch pick."""
    global _last_full
    _last_full = 0.0


def select_child(lod, children, eye):
    """The engine's choice: the highest index whose range holds the distance, or -1 for none."""
    ranges = nif_switch.lod_ranges(lod, children)
    center = lod.matrix_world @ Vector(lod.mw.lod_center)  # rotation, scale, translation, as UpdateWorldData
    scale = lod.matrix_world.to_scale()[0]  # the engine assumes uniform scale
    d2 = (center - eye).length_squared  # fLODAdjust is 1
    for i in reversed(range(len(children))):
        near, far = ranges[i]
        if (near * scale) ** 2 <= d2 < (far * scale) ** 2:  # Python floats do not overflow here
            return i
    return -1


def _depth(ob):
    n = 0
    while ob.parent is not None:
        ob, n = ob.parent, n + 1
    return n


def desired_hidden(scene, view_layer, eye):
    """Names of every object an inactive switch child or LOD level would hide from `eye`."""
    # one pass over the scene: Object.children scans every object in the file on each call
    in_layer = set(view_layer.objects)
    tree = {}
    for ob in nif_switch.scene_objects(scene):
        if ob in in_layer:
            siblings = tree.setdefault(ob.parent, [])
            if ob not in siblings:
                siblings.append(ob)

    def subtree(ob):
        stack = [ob]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(tree.get(current, ()))

    hidden = set()
    nodes = [ob for ob in in_layer if nif_switch.is_switch(ob)]
    for node in sorted(nodes, key=_depth):  # parents first, so a hidden node skips its own choice
        if node.name in hidden:
            _active.pop(node.name, None)
            continue
        children = tree.get(node, [])
        if nif_switch.is_lod(node):
            index = select_child(node, children, eye)
        else:
            index = nif_switch.active_index(node, children)
        _active[node.name] = children[index].name if index >= 0 else None
        for i, child in enumerate(children):
            if i != index:
                hidden.update(ob.name for ob in subtree(child))
    return hidden


def billboard_rotation(mode, world, view):
    """The engine's turn `m` for a billboard (NiBillboardNode::RotateToCamera); its world rotation becomes `W @ m`.

    `world` is the node's world matrix before turning and `view` the camera's world matrix.
    """
    inverse = world.to_3x3().normalized().transposed()
    right = inverse @ view.col[0].xyz  # a Blender view looks down its -Z with +Y up
    up = inverse @ view.col[1].xyz
    forward = inverse @ -view.col[2].xyz
    if mode == 0:  # always face camera: local Z toward the camera, local Y as upright as the view allows
        length = math.hypot(up.y, right.y)
        if length > 1e-6:
            c, s = up.y / length, -right.y / length
            return Matrix((right * c + up * s, up * c - right * s, -forward)).transposed()
    elif mode == 1:  # rotate about up: local Z toward the eye, turning only about local Y
        eye = inverse @ (view.translation - world.translation)
        length = math.hypot(eye.z, eye.x)
        if length >= 1e-12:
            c, s = eye.z / length, eye.x / length
            return Matrix(((c, 0, s), (0, 1, 0), (-s, 0, c)))
    elif mode == 2:  # rigid face camera: local X and Y take the camera's right and up
        # the engine writes +forward, which also mirrors local Z; a delta rotation can only show the turn
        return Matrix((right, up, -forward)).transposed()
    return Matrix.Identity(3)  # a raw value of 3 matches no branch in the engine and does not turn


def _authored_delta(ob, stash):
    euler, quaternion = stash
    if ob.rotation_mode == "QUATERNION":
        return quaternion.normalized().to_matrix()
    euler = euler.copy()
    euler.order = ob.rotation_mode
    return euler.to_matrix()


def _own_rotation(ob):
    if ob.rotation_mode == "QUATERNION":
        return ob.rotation_quaternion.normalized().to_matrix()
    return ob.rotation_euler.to_matrix()


def _stash(ob):
    if not ob.is_property_set("mw_preview_delta"):
        ob.mw_preview_delta = (*ob.delta_rotation_euler, *ob.delta_rotation_quaternion)
        ob.update_tag()  # undo only records objects flagged as changed; every state write must tag
    values = ob.mw_preview_delta
    return Euler(values[:3]), Quaternion(values[3:])


def _unrotate(ob):
    values = ob.mw_preview_delta
    ob.delta_rotation_euler = values[:3]
    ob.delta_rotation_quaternion = values[3:]
    ob.property_unset("mw_preview_delta")
    ob.update_tag()


def unrotate():
    """Write back every billboard's own delta rotation. True if there was any to write."""
    objects = [ob for ob in bpy.data.objects if ob.is_property_set("mw_preview_delta")]
    for ob in objects:
        _unrotate(ob)
    return bool(objects)


def orient(view_layer, view):
    """Turn every billboard in `view_layer` toward `view`, a camera world matrix, through its delta rotation."""
    billboards = [
        ob
        for ob in view_layer.objects
        # Blender has no delta rotation for axis-angle in its Python API
        if nif_switch.node_type(ob) == BILLBOARD and ob.rotation_mode != "AXIS_ANGLE"
    ]
    for ob in bpy.data.objects:
        if ob.is_property_set("mw_preview_delta") and ob not in billboards:
            _unrotate(ob)

    # parents first; a billboard below another reads its parent as of the last evaluation, one tick behind
    for ob in sorted(billboards, key=_depth):
        stash = _stash(ob)
        own = _own_rotation(ob)
        authored = _authored_delta(ob, stash)
        basis = Matrix.LocRotScale(
            ob.location + ob.delta_location,
            authored @ own,
            Vector([a * b for a, b in zip(ob.scale, ob.delta_scale)]),
        )
        # everything above the node, bone parents included, as of the last evaluation
        world = ob.matrix_world @ ob.matrix_basis.inverted_safe() @ basis
        m = billboard_rotation((ob.mw.object_flags >> 5) & 3, world, view)
        # Blender applies the delta before the object's rotation, so conjugate `basis @ m` into it
        delta = authored @ own @ m @ own.transposed()

        current = _authored_delta(ob, (ob.delta_rotation_euler, ob.delta_rotation_quaternion))
        if max(abs(a - b) for row_a, row_b in zip(delta, current) for a, b in zip(row_a, row_b)) < 1e-6:
            continue  # an unchanged write would still tag a redraw
        if ob.rotation_mode == "QUATERNION":
            ob.delta_rotation_quaternion = delta.to_quaternion()
        else:
            ob.delta_rotation_euler = delta.to_euler(ob.rotation_mode, ob.delta_rotation_euler)


def _hide(ob, view_layer):
    ob.hide_set(True, view_layer=view_layer)
    ob.mw_preview_hidden = view_layer.name
    ob.update_tag()


def _unhide(ob):
    for scene in ob.users_scene:
        view_layer = scene.view_layers.get(ob.mw_preview_hidden)
        if view_layer and view_layer.objects.get(ob.name) == ob and ob.hide_get(view_layer=view_layer):
            ob.hide_set(False, view_layer=view_layer)
    ob.property_unset("mw_preview_hidden")
    ob.update_tag()


def apply(view_layer, desired):
    """Hide `desired` and un-hide what this preview hid before but no longer wants hidden."""
    for ob in bpy.data.objects:
        if ob.is_property_set("mw_preview_hidden"):
            if ob.mw_preview_hidden != view_layer.name or ob.name not in desired:
                _unhide(ob)

    for name in desired:
        ob = view_layer.objects.get(name)
        if ob and not ob.hide_get(view_layer=view_layer):  # a new hide, or undo un-hid one of ours
            _hide(ob, view_layer)


def restore():
    """Un-hide everything this preview hid, un-turn its billboards, and forget it."""
    unrotate()
    for ob in bpy.data.objects:
        if ob.is_property_set("mw_preview_hidden"):
            _unhide(ob)
    _active.clear()


def find_eye():
    """The focused window, its view's camera world matrix (or None), and whether any view moved since last call."""
    global _focus
    moved = False
    found = {}
    for wi, window in enumerate(bpy.context.window_manager.windows):
        for ai, area in enumerate(window.screen.areas):
            if area.type != "VIEW_3D":
                continue
            rv3d = area.spaces.active.region_3d
            key = (wi, ai)
            matrix = rv3d.view_matrix.copy()
            found[key] = (window, rv3d)
            if _views.get(key) != matrix:
                if key in _views:
                    _focus = key  # the last view that changed is the one being navigated
                moved = True
                _views[key] = matrix

    for key in list(_views):
        if key not in found:
            del _views[key]
    if not found:
        return None, None, moved
    if _focus not in found:
        _focus = next(iter(found))
    window, rv3d = found[_focus]
    return window, rv3d.view_matrix.inverted(), moved


def _stop():
    global _stopping
    _stopping = True
    try:
        bpy.context.window_manager.mw_node_preview = False
    finally:
        _stopping = False


def tick():
    global _last_full
    try:
        if not bpy.context.window_manager.mw_node_preview:
            restore()
            return None
        window, view, moved = find_eye()
        if view is not None and (moved or time.monotonic() - _last_full > FULL_REFRESH):
            apply(window.view_layer, desired_hidden(window.scene, window.view_layer, view.translation))
            orient(window.view_layer, view)
            _last_full = time.monotonic()
        return INTERVAL
    except Exception:
        traceback.print_exc()  # an uncaught exception would silently unregister the timer
        try:
            restore()
            _stop()  # or the checkbox keeps showing on while nothing runs
        except Exception:
            traceback.print_exc()
        return None


def update_toggle(self, context):
    global _last_full
    if _stopping:
        restore()
        return
    registered = bpy.app.timers.is_registered(tick)
    if self.mw_node_preview:
        _views.clear()
        _last_full = 0.0
        if not registered:
            bpy.app.timers.register(tick, first_interval=0.0, persistent=False)
    else:
        if registered:
            bpy.app.timers.unregister(tick)
        restore()


@persistent
def save_pre(*args):
    restore()  # the file on disk never keeps a preview's hide state or turn


@persistent
def save_post(*args):
    refresh()  # the next tick hides inactive children and turns the billboards again


@persistent
def load_pre(*args):
    global _focus
    _views.clear()
    _active.clear()
    _focus = None


@persistent
def load_post(*args):
    restore()  # clipboard, autosave and library writes can bypass save_pre
    wm = bpy.context.window_manager
    if wm is not None and wm.mw_node_preview:
        wm.mw_node_preview = False


@persistent
def history_post(*args):
    wm = bpy.context.window_manager
    if wm is not None and wm.mw_node_preview:
        refresh()  # the step may hold another moment's turn and hides; the next tick re-evaluates
    else:
        restore()  # a step recorded while previewing brought its turn and hides back


_handlers = (
    (bpy.app.handlers.save_pre, save_pre),
    (bpy.app.handlers.save_post, save_post),
    (bpy.app.handlers.load_pre, load_pre),
    (bpy.app.handlers.load_post, load_post),
    (bpy.app.handlers.undo_post, history_post),
    (bpy.app.handlers.redo_post, history_post),
)


def register():
    bpy.types.WindowManager.mw_node_preview = bpy.props.BoolProperty(
        name="Game Preview",
        description="Hide the inactive children of switches and LODs and turn billboards, as the game would "
        "for the current view. Not saved",
        default=False,
        update=update_toggle,
    )
    bpy.types.Object.mw_preview_delta = bpy.props.FloatVectorProperty(size=7, options={"HIDDEN", "SKIP_SAVE"})
    bpy.types.Object.mw_preview_hidden = bpy.props.StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    for handlers, handler in _handlers:
        if handler not in handlers:
            handlers.append(handler)


def unregister():
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)
    restore()
    for handlers, handler in _handlers:
        if handler in handlers:
            handlers.remove(handler)
    del bpy.types.WindowManager.mw_node_preview
    del bpy.types.Object.mw_preview_delta
    del bpy.types.Object.mw_preview_hidden
