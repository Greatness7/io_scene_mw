from __future__ import annotations

import contextlib
import pathlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, cast

import bpy


ADDON_ID = "io_scene_mw"


class _TexturePath(Protocol):
    name: str


class _Preferences(Protocol):
    scale_correction: float
    texture_paths: list[_TexturePath]


@dataclass(frozen=True, slots=True)
class ImportRuntime:
    context: bpy.types.Context
    scene: bpy.types.Scene
    view_layer: bpy.types.ViewLayer
    active_collection: bpy.types.Collection
    scale_correction: float
    fps: float
    texture_paths: tuple[pathlib.Path, ...]

    @classmethod
    def from_context(cls, context: bpy.types.Context) -> ImportRuntime:
        scene = context.scene
        view_layer = context.view_layer
        preferences_root = context.preferences
        if scene is None or view_layer is None or preferences_root is None:
            raise RuntimeError("NIF import requires an active scene, view layer, and preferences")
        active_layer_collection = view_layer.active_layer_collection
        if active_layer_collection is None:
            raise RuntimeError("NIF import requires an active collection")
        preferences = cast(_Preferences, preferences_root.addons[ADDON_ID].preferences)
        return cls(
            context=context,
            scene=scene,
            view_layer=view_layer,
            active_collection=active_layer_collection.collection,
            scale_correction=preferences.scale_correction,
            fps=scene.render.fps,
            texture_paths=tuple(pathlib.Path(item.name) for item in preferences.texture_paths),
        )


@dataclass(frozen=True, slots=True)
class ExportRuntime:
    context: bpy.types.Context
    scene: bpy.types.Scene
    view_layer: bpy.types.ViewLayer
    scale_correction: float
    fps: float
    frame_start: float
    frame_end: float

    @classmethod
    def from_context(cls, context: bpy.types.Context) -> ExportRuntime:
        scene = context.scene
        view_layer = context.view_layer
        preferences_root = context.preferences
        if scene is None or view_layer is None or preferences_root is None:
            raise RuntimeError("NIF export requires an active scene, view layer, and preferences")
        preferences = cast(_Preferences, preferences_root.addons[ADDON_ID].preferences)
        return cls(
            context=context,
            scene=scene,
            view_layer=view_layer,
            scale_correction=1 / preferences.scale_correction,
            fps=scene.render.fps,
            frame_start=scene.frame_start,
            frame_end=scene.frame_end,
        )


def require_object_mode(context: bpy.types.Context) -> None:
    if context.mode != "OBJECT":
        raise RuntimeError(f"NIF import requires Object Mode, but Blender is in {context.mode} mode")


@contextlib.contextmanager
def scene_frame_scope(scene: bpy.types.Scene, frame: int = 0) -> Iterator[None]:
    """Evaluate the scene at `frame`, then return to the current frame, even on failure.

    Only the current frame is restored: the frame range an import widens stays widened.
    """
    saved = scene.frame_current
    scene.frame_set(frame)
    try:
        yield
    finally:
        scene.frame_set(saved)


@contextlib.contextmanager
def object_edit_scope(runtime: ImportRuntime, obj: Any) -> Iterator[None]:
    """Make `obj` active in Edit Mode, and return to Object Mode on exit, even on failure.

    The previously active object is not restored; import makes its root active once it finishes.
    """
    require_object_mode(runtime.context)
    runtime.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        yield
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


@contextlib.contextmanager
def export_evaluation_scope(runtime: ExportRuntime, sources: Iterable[Any]) -> Iterator[Any]:
    """Put mesh objects in their export evaluation state and yield a dependency graph updated for it.

    While the scope is active each source is visible, undeformed by armatures, triangulated, and has every shape
    key muted, so evaluation yields the triangulated rest mesh. Objects and shared shape-key datablocks are
    changed once however often they occur, and on exit every changed value is restored and every added modifier
    removed. The graph must not be read after the scope exits.
    """
    objects = list({source.as_pointer(): source for source in sources}.values())
    unhidden: list[Any] = []
    disabled: list[Any] = []
    added: list[tuple[Any, Any]] = []
    muted: list[Any] = []
    try:
        muted_keys: set[int] = set()
        for source in objects:
            # hidden switch variants still export, but hidden objects are not evaluated
            if source.hide_viewport:
                source.hide_viewport = False
                unhidden.append(source)

            visible = [modifier for modifier in source.modifiers if modifier.show_viewport]
            for modifier in visible:
                if modifier.type == "ARMATURE":
                    modifier.show_viewport = False
                    disabled.append(modifier)

            if not any(modifier.type == "TRIANGULATE" for modifier in visible):
                modifier = source.modifiers.new("", "TRIANGULATE")
                added.append((source, modifier))
                modifier.quad_method = "FIXED"
                modifier.ngon_method = "CLIP"
                if bpy.app.version < (4, 2, 0):
                    modifier.keep_custom_normals = True

            shape_keys = source.data.shape_keys
            if shape_keys is not None and shape_keys.as_pointer() not in muted_keys:
                muted_keys.add(shape_keys.as_pointer())
                for block in shape_keys.key_blocks:
                    if not block.mute:
                        block.mute = True
                        muted.append(block)

        depsgraph = runtime.context.evaluated_depsgraph_get()
        depsgraph.update()
        yield depsgraph
    finally:
        for block in reversed(muted):
            block.mute = False
        for source, modifier in reversed(added):
            source.modifiers.remove(modifier)
        for modifier in reversed(disabled):
            modifier.show_viewport = True
        for source in reversed(unhidden):
            source.hide_viewport = True


def acquire_evaluated_mesh(source: Any, depsgraph: Any) -> tuple[Any, Any]:
    """The evaluated object of `source` and a temporary mesh that `release_evaluated_mesh` must free."""
    evaluated = source.evaluated_get(depsgraph)
    return evaluated, evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)


def release_evaluated_mesh(evaluated: Any) -> None:
    evaluated.to_mesh_clear()


@contextlib.contextmanager
def evaluated_mesh(source: Any, depsgraph: Any) -> Iterator[tuple[Any, Any]]:
    """Yield the evaluated object and temporary mesh of `source`; the mesh is freed on exit, even on failure."""
    evaluated, mesh = acquire_evaluated_mesh(source, depsgraph)
    try:
        yield evaluated, mesh
    finally:
        release_evaluated_mesh(evaluated)
