"""Live viewport preview of NiSwitchNode and NiLODNode selections.

A timer reads the navigated 3D view's eye position, applies the engine's selection rule to every switch and
LOD Empty, and hides the subtrees of inactive children with `hide_set`. It only ever un-hides what it hid itself, and the
file on disk never keeps a preview's hide state. All state here is runtime-only.
"""

import time
import traceback

import bpy
from bpy.app.handlers import persistent
from mathutils import Vector

from . import nif_switch

INTERVAL = 0.1
FULL_REFRESH = 1.0  # also re-evaluate a still view, to pick up range edits and undo

_hidden = set()  # names of the objects this preview hid
_layer = None  # (scene name, view layer name) that `_hidden` applies to
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
    centre = lod.matrix_world @ Vector(lod.mw.lod_center)  # rotation, scale, translation, as UpdateWorldData
    scale = lod.matrix_world.to_scale()[0]  # the engine assumes uniform scale
    d2 = (centre - eye).length_squared  # fLODAdjust is 1
    for i in reversed(range(len(children))):
        near, far = ranges[i]
        if (near * scale) ** 2 <= d2 < (far * scale) ** 2:  # Python floats do not overflow here
            return i
    return -1


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

    def depth(ob):
        n = 0
        while ob.parent is not None:
            ob, n = ob.parent, n + 1
        return n

    def subtree(ob):
        stack = [ob]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(tree.get(current, ()))

    hidden = set()
    nodes = [ob for ob in in_layer if nif_switch.is_switch(ob)]
    for node in sorted(nodes, key=depth):  # parents first, so a hidden node skips its own choice
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


def _get_layer(key):
    scene = bpy.data.scenes.get(key[0]) if key else None
    return scene.view_layers.get(key[1]) if scene else None


def apply(view_layer, desired):
    """Hide `desired` and un-hide what this preview hid before but no longer wants hidden."""
    global _layer
    key = (view_layer.id_data.name, view_layer.name)
    if _layer != key:
        _set_hidden(False)  # un-hide in the old layer; `_active` is already this layer's
        _hidden.clear()
        _layer = key
    for name in desired:
        ob = bpy.data.objects.get(name)  # by name every time: undo invalidates held references
        if ob and not ob.hide_get(view_layer=view_layer):  # a new hide, or undo un-hid one of ours
            ob.hide_set(True, view_layer=view_layer)
            _hidden.add(name)
    for name in list(_hidden - desired):
        ob = bpy.data.objects.get(name)
        if ob and ob.hide_get(view_layer=view_layer):
            ob.hide_set(False, view_layer=view_layer)
        _hidden.discard(name)


def _set_hidden(state):
    view_layer = _get_layer(_layer)
    if view_layer is None:
        return
    for name in _hidden:
        ob = bpy.data.objects.get(name)
        if ob and view_layer.objects.get(name) and ob.hide_get(view_layer=view_layer) != state:
            ob.hide_set(state, view_layer=view_layer)


def restore():
    """Un-hide everything this preview hid and forget it."""
    global _layer
    _set_hidden(False)
    _hidden.clear()
    _active.clear()
    _layer = None


def find_eye():
    """The focused window, its view's eye position (or None), and whether any view moved since last call."""
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
    return window, rv3d.view_matrix.inverted().translation, moved


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
        window, eye, moved = find_eye()
        if eye is not None and (moved or time.monotonic() - _last_full > FULL_REFRESH):
            apply(window.view_layer, desired_hidden(window.scene, window.view_layer, eye))
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
    _set_hidden(False)  # the file on disk never keeps a preview's hide state


@persistent
def save_post(*args):
    _set_hidden(True)


@persistent
def load_pre(*args):
    global _layer, _focus
    _hidden.clear()  # the names belong to the old file
    _views.clear()
    _active.clear()
    _layer = _focus = None


@persistent
def load_post(*args):
    wm = bpy.context.window_manager
    if wm is not None and wm.mw_node_preview:
        wm.mw_node_preview = False


_handlers = (
    (bpy.app.handlers.save_pre, save_pre),
    (bpy.app.handlers.save_post, save_post),
    (bpy.app.handlers.load_pre, load_pre),
    (bpy.app.handlers.load_post, load_post),
)


def register():
    bpy.types.WindowManager.mw_node_preview = bpy.props.BoolProperty(
        name="Enable Preview",
        description="Hide inactive switch children, and LOD levels the way the game would at the current "
        "viewport distance. Not saved",
        default=False,
        update=update_toggle,
    )
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
