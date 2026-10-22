from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import bpy
import numpy as np
import numpy.linalg as la
from es3 import nif
from es3.utils.math import ID44, decompose, quaternion_from_matrix, quaternion_mul

from .blender_compat import fcurves_by_data_path
from .options import ExportOptions
from .records import Array


def collect_keyframe_points(fcurves, key_type, dtype=np.float32):
    num_axes = len(fcurves)
    num_frames = len(fcurves[0].keyframe_points)
    num_values = 1 + num_axes * (1 if key_type.name == "LIN_KEY" else 3)
    keys = np.empty((num_frames, num_values), dtype)
    temp = np.empty((num_frames, 2), dtype)
    for i, fcurve in enumerate(fcurves, start=1):
        fcurve.keyframe_points.foreach_get("co", temp.ravel())
        keys[:, (0, i)] = temp
        if key_type.name == "BEZ_KEY":
            fcurve.keyframe_points.foreach_get("handle_left", temp.ravel())
            keys[:, i + num_axes] = (keys[:, i] - temp[:, 1]) * 3.0
            fcurve.keyframe_points.foreach_get("handle_right", temp.ravel())
            keys[:, i + num_axes * 2] = (temp[:, 1] - keys[:, i]) * 3.0
    return keys[keys[:, 0].argsort()]


def get_interpolation_type(fcurves):
    try:
        interpolation = fcurves[0].keyframe_points[0].interpolation
    except IndexError:
        return None
    if interpolation == "LINEAR":
        return nif.NiFloatData.KeyType.LIN_KEY
    return nif.NiFloatData.KeyType.BEZ_KEY


def create_uv_controller(bl_prop, bl_slot, source_object, target_shape, options: ExportOptions, cycle_type: str):
    if not options.export_animations:
        return False
    if not (bl_slot.image and bl_slot.layer):
        return False
    anims = fcurves_by_data_path(bl_prop.texture_group.node_tree)
    if len(anims) == 0:
        return False
    uv_data = nif.NiUVData()
    bl_node = bl_slot.mapping_node
    channels = {
        (uv_data.u_offset_data, uv_data.v_offset_data): bl_node.inputs["Location"].path_from_id("default_value"),
        (uv_data.u_tiling_data, uv_data.v_tiling_data): bl_node.inputs["Scale"].path_from_id("default_value"),
    }
    if not any(map(anims.get, channels.values())):
        return False
    for outputs, data_path in channels.items():
        for i, output in enumerate(outputs):
            fcurves = [fcurve for fcurve in anims[data_path] if fcurve.array_index == i]
            key_type = get_interpolation_type(fcurves)
            if key_type is None:
                continue
            output.key_type = key_type
            output.keys = collect_keyframe_points(fcurves, key_type)
    try:
        uv_data.u_offset_data.keys[:, 1] *= -1
        uv_data.v_offset_data.keys[:, 1] *= -1
    except AttributeError:
        pass
    controller = nif.NiUVController(cycle_type=cycle_type, target=target_shape, data=uv_data)
    controller.update_start_stop_times()
    for i, uv in enumerate(source_object.data.uv_layers):
        if uv.name == bl_slot.layer:
            controller.texture_set = i
            break
    target_shape.controllers.appendleft(controller)
    return True


def create_material_controllers(ni_prop, bl_prop, options: ExportOptions, cycle_type: str) -> None:
    if not options.export_animations:
        return
    anims = fcurves_by_data_path(bl_prop.material.node_tree)
    if len(anims) == 0:
        return
    create_color_controller(anims, ni_prop, bl_prop, cycle_type)
    create_alpha_controller(anims, ni_prop, bl_prop, cycle_type)


def create_color_controller(fcurves_dict, ni_prop, bl_prop, cycle_type: str):
    channels = {"DIFFUSE": bl_prop.diffuse_input, "EMISSIVE": bl_prop.emissive_input}
    for color_field, source in channels.items():
        data_path = source.path_from_id("default_value")
        fcurves = fcurves_dict[data_path]
        key_type = get_interpolation_type(fcurves)
        if key_type is None:
            continue
        keys = collect_keyframe_points(fcurves[:3], key_type)
        if len(keys) == 0:
            continue
        controller = nif.NiMaterialColorController(
            cycle_type=cycle_type,
            color_field=color_field,
            target=ni_prop,
            data=nif.NiPosData(key_type=key_type, keys=keys),
        )
        controller.update_start_stop_times()
        ni_prop.controllers.appendleft(controller)
    return True


def create_alpha_controller(fcurves_dict, ni_prop, bl_prop, cycle_type: str):
    data_path = bl_prop.opacity_input.path_from_id("default_value")
    fcurves = fcurves_dict[data_path]
    key_type = get_interpolation_type(fcurves)
    if key_type is None:
        return False
    keys = collect_keyframe_points(fcurves, key_type)
    if len(keys) == 0:
        return False
    controller = nif.NiAlphaController(
        cycle_type=cycle_type,
        target=ni_prop,
        data=nif.NiFloatData(key_type=key_type, keys=keys),
    )
    controller.update_start_stop_times()
    ni_prop.controllers.appendleft(controller)
    return True


def export_object_animation(
    source: bpy.types.Object | bpy.types.PoseBone,
    name: str,
    target: nif.NiAVObject,
    parent_world: Array | None,
    axis_correction: Array,
    options: ExportOptions,
    cycle_type: str,
    warn: Callable[[str], None],
) -> None:
    _ObjectAnimation(source, name, target, parent_world, axis_correction, options, cycle_type, warn).create()


@dataclass(frozen=True, slots=True)
class _ObjectAnimation:
    source: bpy.types.Object | bpy.types.PoseBone
    name: str
    target: nif.NiAVObject
    parent_world: Array | None
    axis_correction: Array
    options: ExportOptions
    cycle_type: str
    warn: Callable[[str], None]

    def create(self) -> None:
        if not self.options.export_animations:
            return
        self.create_text_keys()
        fcurves_dict = fcurves_by_data_path(self.source.id_data)
        if len(fcurves_dict) == 0:
            return
        self.create_vis_controller(fcurves_dict)
        self.create_translations(fcurves_dict)
        self.create_rotations(fcurves_dict)
        self.create_scales(fcurves_dict)

    def create_text_keys(self):
        try:
            markers = cast(Any, self.source).animation_data.action.pose_markers
        except AttributeError:
            return False
        if len(markers) == 0:
            return False
        text_data = nif.NiTextKeyExtraData()
        text_data.keys.resize(len(markers))
        for i, marker in enumerate(markers):
            text_data.keys[i] = marker.frame, marker.name.replace("; ", "\r\n")
        text_data.collapse_groups()
        text_data.keys.sort()
        self.target.extra_data = text_data
        return True

    def create_vis_controller(self, fcurves_dict):
        key_type = nif.NiFloatData.KeyType.LIN_KEY
        _, keys = self.collect_keys(fcurves_dict, "hide_viewport", num_axes=1, key_type=key_type)
        if len(keys) == 0:
            return False
        controller = nif.NiVisController(
            cycle_type=self.cycle_type,
            target=self.target,
            data=nif.NiVisData(),
        )
        data = controller.data
        assert data is not None
        data.keys.resize(len(keys))
        data.times = keys[:, 0]
        data.values = 1 - keys[:, 1]
        controller.update_start_stop_times()
        self.target.controllers.appendleft(controller)
        return True

    def create_translations(self, fcurves_dict):
        key_type, keys = self.collect_keys(fcurves_dict, "location", num_axes=3)
        if len(keys) == 0:
            return False
        assert key_type is not None
        controller = self.create_keyframe_controller()
        data = controller.data
        assert data is not None
        data.translations.key_type = key_type
        data.translations.keys = keys
        controller.start_time = min(keys[0, 0], controller.start_time)
        controller.stop_time = max(keys[-1, 0], controller.stop_time)
        offset = self.to_local_space()
        if not np.allclose(offset, ID44, rtol=0, atol=1e-4):
            translations = data.translations
            translations.values[:] = translations.values @ offset[:3, :3].T + offset[:3, 3]
            if key_type.name == "BEZ_KEY":
                translations.in_tans[:] = translations.in_tans @ offset[:3, :3].T
                translations.out_tans[:] = translations.out_tans @ offset[:3, :3].T
        return True

    def create_rotations(self, fcurves_dict):
        has_euler = self.create_euler_rotations(fcurves_dict)
        has_quats = self.create_quaternion_rotations(fcurves_dict)
        if has_euler and has_quats:
            raise ValueError(f"'({self.name})' mixing euler and quaternion rotations in the same action is not supported")
        return has_euler or has_quats

    def create_euler_rotations(self, fcurves_dict):
        key_type, keys = self.collect_keys(fcurves_dict, "rotation_euler", num_axes=3)
        if len(keys) == 0:
            return False
        assert key_type is not None
        if isinstance(self.source, bpy.types.PoseBone):
            raise ValueError(f"({self.name}) bones euler animations are not currently supported.")
        controller = self.create_keyframe_controller()
        data = controller.data
        assert data is not None
        rotations = data.rotations
        rotations.key_type = rotations.KeyType.EULER_KEY
        if key_type.name == "LIN_KEY":
            axis_columns = [(0, 1 + i) for i in range(3)]
        else:
            axis_columns = [(0, 1 + i, 4 + i, 7 + i) for i in range(3)]
        rotations.euler_data = tuple(nif.NiFloatData(key_type=key_type, keys=keys[:, columns]) for columns in axis_columns)
        controller.start_time = min(keys[0, 0], controller.start_time)
        controller.stop_time = max(keys[-1, 0], controller.stop_time)
        offset = self.to_local_space()
        if not np.allclose(offset, ID44, rtol=0, atol=1e-4):
            rotations.convert_to_quaternions()
            values = data.rotations.values
            offset = quaternion_from_matrix(offset)
            quaternion_mul(offset, values, out=values)
            values /= la.norm(values, axis=1)[:, None]
        return True

    def create_quaternion_rotations(self, fcurves_dict):
        key_type = nif.NiRotData.KeyType.LIN_KEY
        _, keys = self.collect_keys(
            fcurves_dict,
            "rotation_quaternion",
            num_axes=4,
            key_type=key_type,
            sample=True,
        )
        if len(keys) == 0:
            return False
        controller = self.create_keyframe_controller()
        data = controller.data
        assert data is not None
        values = keys[:, 1:]
        if isinstance(self.source, bpy.types.PoseBone):
            offset = quaternion_from_matrix(self.axis_correction)
            quaternion_mul(values, offset, out=values)
        offset = self.to_local_space()
        if not np.allclose(offset, ID44, rtol=0, atol=1e-4):
            offset = quaternion_from_matrix(offset)
            quaternion_mul(offset, values, out=values)
        values /= la.norm(values, axis=1)[:, None]
        data.rotations.keys = keys
        controller.start_time = min(keys[0, 0], controller.start_time)
        controller.stop_time = max(keys[-1, 0], controller.stop_time)
        return True

    def create_scales(self, fcurves_dict):
        key_type, keys = self.collect_keys(fcurves_dict, "scale", num_axes=3)
        if len(keys) == 0:
            return False
        assert key_type is not None
        scales_min = keys[:, 1:4].min(axis=1)
        scales_max = keys[:, 1:4].max(axis=1)
        if not np.allclose(scales_min, scales_max, rtol=0, atol=1e-4):
            self.warn(f"({self.name}) non-uniform scale animations are not supported")
            return False
        if key_type.name == "LIN_KEY":
            keys = keys[:, :2]
        else:
            keys = keys[:, (0, 1, 4, 7)]
        controller = self.create_keyframe_controller()
        data = controller.data
        assert data is not None
        data.scales.key_type = key_type
        data.scales.keys = keys
        controller.start_time = min(keys[0, 0], controller.start_time)
        controller.stop_time = max(keys[-1, 0], controller.stop_time)
        offset = self.to_local_space()
        if not np.allclose(offset, ID44, rtol=0, atol=1e-4):
            offset_scale = decompose(offset)[-1][-1]
            scales = data.scales
            scales.values[:] *= offset_scale
            if key_type.name == "BEZ_KEY":
                scales.in_tans[:] *= offset_scale
                scales.out_tans[:] *= offset_scale
        return True

    def to_local_space(self) -> Array:
        if isinstance(self.source, bpy.types.PoseBone):
            matrix = self.source.id_data.convert_space(
                pose_bone=self.source,
                matrix=ID44.T,
                from_space="LOCAL",
                to_space="WORLD",
            )
            assert self.parent_world is not None
            return cast(Array, la.solve(self.parent_world, matrix))
        if self.parent_world is not None:
            # keys live in blender's parent frame (parent or bone, then the parent inverse), which moves with the nif parent
            parent_frame = np.asarray(self.source.matrix_world) @ la.inv(np.asarray(self.source.matrix_basis))
            return cast(Array, la.solve(self.parent_world, parent_frame))
        return ID44

    def collect_keys(self, fcurves_dict, key, num_axes, key_type=None, sample=False):
        if isinstance(self.source, bpy.types.PoseBone):
            data_path = f'pose.bones["{self.name}"].{key}'
        else:
            data_path = self.source.path_from_id(key)
        fcurves = {fcurve.array_index: fcurve for fcurve in fcurves_dict.get(data_path, ()) if len(fcurve.keyframe_points)}
        if len(fcurves) == 0:
            return None, ()
        if key_type is None:
            key_type = get_interpolation_type([fcurves[i] for i in sorted(fcurves)])
        assert key_type is not None
        axes = {
            i: nif.NiFloatData(key_type=key_type, keys=collect_keyframe_points([fcurve], key_type))
            for i, fcurve in fcurves.items()
        }
        times = np.unique(np.concatenate([axis.times for axis in axes.values()]))
        times = times[np.concatenate(([True], np.diff(times) > 1e-6))]
        values, in_tans, out_tans = [], [], []
        for i in range(num_axes):
            axis = axes.get(i)
            if axis is None:
                static = getattr(self.source, key)[i]
                values.append(np.full(len(times), static))
                in_tans.append(np.zeros(len(times)))
                out_tans.append(np.zeros(len(times)))
                continue
            if sample:
                values.append([fcurves[i].evaluate(time) for time in times])
                continue
            axis.evaluate_and_insert(times)
            assert len(axis.keys) == len(times)
            values.append(axis.values)
            if key_type.name == "BEZ_KEY":
                in_tans.append(axis.in_tans)
                out_tans.append(axis.out_tans)
        columns = [times, *values]
        if key_type.name == "BEZ_KEY":
            columns += [*in_tans, *out_tans]
        return key_type, np.column_stack(columns).astype(np.float32)

    def create_keyframe_controller(self) -> nif.NiKeyframeController:
        result = self.target.controllers.find_type_with_owner(nif.NiKeyframeController)
        if result is not None:
            _, controller = result
        else:
            controller = nif.NiKeyframeController(
                cycle_type=self.cycle_type,
                target=self.target,
                data=nif.NiKeyframeData(),
            )
            self.target.controllers.appendleft(controller)
        return controller
