from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Protocol, cast

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
