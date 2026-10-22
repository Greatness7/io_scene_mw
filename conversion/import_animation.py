from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import bpy
import numpy as np
from es3 import nif

from .blender_compat import action_fcurves, ensure_action, ensure_fcurve, set_fcurves_mute
from .options import ImportOptions


@dataclass(frozen=True, slots=True)
class ImportAnimationContext:
    options: ImportOptions
    scene: bpy.types.Scene
    warn: Callable[[str], None]


def create_interpolation_data(ni_data, fcurve, axis: Any = ...) -> None:
    if ni_data.key_type.name == "NO_INTERP":
        for point in fcurve.keyframe_points:
            point.interpolation = "CONSTANT"
    elif ni_data.key_type.name == "LIN_KEY":
        for point in fcurve.keyframe_points:
            point.interpolation = "LINEAR"
    else:
        for point in fcurve.keyframe_points:
            point.interpolation = "BEZIER"
            point.handle_left_type = point.handle_right_type = "FREE"
        handles = ni_data.get_tangent_handles()
        fcurve.keyframe_points.foreach_set("handle_left", handles[0, axis].ravel())
        fcurve.keyframe_points.foreach_set("handle_right", handles[1, axis].ravel())


def update_frame_range(scene, controller) -> None:
    frame_end = int(np.ceil(controller.stop_time))
    scene.frame_end = scene.frame_preview_end = max(scene.frame_end, frame_end)


def create_uv_controller(controller, target_object, ctx: ImportAnimationContext) -> None:
    if ctx.options.ignore_animations:
        return
    data = controller.data
    if data is None:
        return
    try:
        bl_prop = target_object.active_material.mw
    except AttributeError:
        return
    action = ensure_action(bl_prop.texture_group.node_tree)
    try:
        uv_name = target_object.data.uv_layers[controller.texture_set].name
        bl_slot = next(slot for slot in bl_prop.texture_slots if slot.layer == uv_name)
        bl_node = bl_slot.mapping_node
    except (IndexError, StopIteration):
        ctx.warn("skipping NiUVController due to invalid texture set")
        return
    channels = {
        (data.u_offset_data, data.v_offset_data): (
            bl_node.inputs["Location"].path_from_id("default_value"),
            True,
        ),
        (data.u_tiling_data, data.v_tiling_data): (
            bl_node.inputs["Scale"].path_from_id("default_value"),
            False,
        ),
    }
    for sources, (data_path, negate) in channels.items():
        for i, uv_data in enumerate(sources):
            curve_data = nif.NiFloatData(key_type=uv_data.key_type, keys=uv_data.keys.copy())
            if negate:
                curve_data.keys[:, 1] *= -1
            fcurve = ensure_fcurve(action, data_path, index=i, group=uv_name)
            fcurve.keyframe_points.add(len(uv_data.keys))
            fcurve.keyframe_points.foreach_set("co", curve_data.keys[:, :2].ravel())
            create_interpolation_data(curve_data, fcurve)
            fcurve.update()
    update_frame_range(ctx.scene, controller)


def create_color_controller(bl_prop, ni_prop, ctx: ImportAnimationContext) -> None:
    if ctx.options.ignore_animations:
        return
    controller = ni_prop.controllers.find_type(nif.NiMaterialColorController)
    if controller is None or controller.data is None:
        return
    keys = controller.data.keys
    if len(keys) == 0:
        return
    if controller.color_field == "DIFFUSE":
        data_path = bl_prop.diffuse_input.path_from_id("default_value")
    elif controller.color_field == "EMISSIVE":
        data_path = bl_prop.emissive_input.path_from_id("default_value")
    else:
        raise NotImplementedError(f"'{controller.color_field}' animations are not supported")
    action = ensure_action(bl_prop.material.node_tree)
    for i in range(3):
        fcurve = ensure_fcurve(action, data_path, index=i, group=bl_prop.material.name)
        fcurve.keyframe_points.add(len(keys))
        fcurve.keyframe_points.foreach_set("co", keys[:, (0, i + 1)].ravel())
        create_interpolation_data(controller.data, fcurve)
        fcurve.update()
    update_frame_range(ctx.scene, controller)


def create_alpha_controller(bl_prop, ni_prop, ctx: ImportAnimationContext) -> None:
    if ctx.options.ignore_animations:
        return
    controller = ni_prop.controllers.find_type(nif.NiAlphaController)
    if controller is None or controller.data is None:
        return
    keys = controller.data.keys
    if len(keys) == 0:
        return
    action = ensure_action(bl_prop.material.node_tree)
    data_path = bl_prop.opacity_input.path_from_id("default_value")
    fcurve = ensure_fcurve(action, data_path, index=0, group=bl_prop.material.name)
    fcurve.keyframe_points.add(len(keys))
    fcurve.keyframe_points.foreach_set("co", keys[:, :2].ravel())
    create_interpolation_data(controller.data, fcurve)
    fcurve.update()
    update_frame_range(ctx.scene, controller)


def import_object_animation(source_node, target, bone_name: str, ctx: ImportAnimationContext) -> None:
    _ObjectAnimation(source_node, target, bone_name, ctx).create()


def set_mute(target, state: bool, ctx: ImportAnimationContext, fcurves=None) -> None:
    if ctx.options.ignore_animations:
        return
    if fcurves is None:
        try:
            action = target.id_data.animation_data.action
        except AttributeError:
            return
        if action is None:
            return
        fcurves = action_fcurves(action)
    if not fcurves:
        return
    set_fcurves_mute(fcurves, state)


@dataclass(frozen=True, slots=True)
class _ObjectAnimation:
    source: nif.NiAVObject
    target: bpy.types.Object | bpy.types.PoseBone
    bone_name: str
    ctx: ImportAnimationContext

    def create(self) -> None:
        if self.ctx.options.ignore_animations:
            return
        bl_object = self.target.id_data
        if self.source.extra_data:
            self.create_text_keys(bl_object)
        if self.source.controller:
            self.create_kf_controller(bl_object)
            self.create_vis_controller(bl_object)

    def create_text_keys(self, bl_object) -> None:
        text_data = self.source.extra_datas.find_type(nif.NiTextKeyExtraData)
        if text_data is None:
            return
        action = ensure_action(bl_object)
        for frame, text in text_data.keys.tolist():
            for name in filter(None, text.splitlines()):
                assert len(name) < 64, f"Marker exceeds character limit ({name})"
                action.pose_markers.new(name).frame = round(frame)

    def create_kf_controller(self, bl_object) -> None:
        controller = self.source.controllers.find_type(nif.NiKeyframeController)
        if not (controller and controller.data):
            return
        action = ensure_action(bl_object)
        self.create_translations(controller, action)
        self.create_rotations(controller, action)
        self.create_scales(controller, action)
        update_frame_range(self.ctx.scene, controller)

    def create_translations(self, controller, action) -> None:
        data = controller.data.translations
        if len(data.keys) == 0:
            return
        data_path = self.target.path_from_id("location")
        for i in range(3):
            fcurve = ensure_fcurve(action, data_path, index=i, group=self.bone_name)
            fcurve.keyframe_points.add(len(data.keys))
            fcurve.keyframe_points.foreach_set("co", data.keys[:, (0, i + 1)].ravel())
            create_interpolation_data(data, fcurve, axis=i)
            fcurve.update()

    def create_rotations(self, controller, action) -> None:
        if controller.data.rotations.euler_data:
            if isinstance(self.target, bpy.types.PoseBone):
                print(f"[INFO] Euler animations on bones are not currently supported. ({self.source.name})")
                controller.data.rotations.convert_to_quaternions()
            else:
                self.target.rotation_mode = controller.data.rotations.euler_axis_order.name
                self.create_euler_rotations(controller, action)
                return
        self.target.rotation_mode = "QUATERNION"
        self.create_quaternion_rotations(controller, action)

    def create_euler_rotations(self, controller, action) -> None:
        for i, data in enumerate(controller.data.rotations.euler_data):
            if len(data.keys) == 0:
                continue
            data_path = self.target.path_from_id("rotation_euler")
            fcurve = ensure_fcurve(action, data_path, index=i, group=self.target.name)
            fcurve.keyframe_points.add(len(data.keys))
            fcurve.keyframe_points.foreach_set("co", data.keys[:, :2].ravel())
            create_interpolation_data(data, fcurve)
            fcurve.update()

    def create_quaternion_rotations(self, controller, action) -> None:
        data = controller.data.rotations
        if len(data.keys) == 0:
            return
        data_path = self.target.path_from_id("rotation_quaternion")
        for i in range(4):
            fcurve = ensure_fcurve(action, data_path, index=i, group=self.target.name)
            fcurve.keyframe_points.add(len(data.keys))
            fcurve.keyframe_points.foreach_set("co", data.keys[:, (0, i + 1)].ravel())
            fcurve.update()

    def create_scales(self, controller, action) -> None:
        data = controller.data.scales
        if len(data.keys) == 0:
            return
        data_path = self.target.path_from_id("scale")
        for i in range(3):
            fcurve = ensure_fcurve(action, data_path, index=i, group=self.target.name)
            fcurve.keyframe_points.add(len(data.keys))
            fcurve.keyframe_points.foreach_set("co", data.keys[:, :2].ravel())
            create_interpolation_data(data, fcurve)
            fcurve.update()

    def create_vis_controller(self, bl_object) -> None:
        controller = self.source.controllers.find_type(nif.NiVisController)
        if controller is None or controller.data is None or len(controller.data.keys) == 0:
            return
        data = controller.data
        keys = np.empty((len(data.keys), 2), dtype=np.float32)
        keys[:, 0] = data.times
        keys[:, 1] = 1 - data.values
        action = ensure_action(bl_object)
        try:
            data_path = self.target.path_from_id("hide_viewport")
        except AttributeError:
            self.ctx.warn(f"NiVisController on bones are not supported ({self.source.name})")
            return
        fcurve = ensure_fcurve(action, data_path, index=0, group=self.target.name)
        fcurve.keyframe_points.add(len(keys))
        fcurve.keyframe_points.foreach_set("co", keys.ravel())
        fcurve.update()
