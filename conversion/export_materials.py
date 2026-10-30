from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, cast

import bpy
import numpy as np
from es3 import nif

from .. import nif_shader, nif_switch
from . import export_animation
from .options import ExportOptions


@dataclass(frozen=True, slots=True)
class ExportMaterialContext:
    options: ExportOptions
    cycle_type: str
    materials: dict[bpy.types.Material, list[nif.NiProperty | None]]
    pixel_datas: dict[bpy.types.Image, nif.NiPixelData]


def export_material(material, target_shape, source_object, ctx: ExportMaterialContext, *, is_collider):
    if is_collider:
        return
    if not (material and material.use_nodes):
        return
    try:
        bl_prop = material.mw.validate()
    except TypeError:
        bl_prop = None
    try:
        if not cast(Any, bl_prop).use_vertex_colors:
            del target_shape.data.vertex_colors
    except AttributeError:
        pass
    if material in ctx.materials:
        target_shape.properties = list(ctx.materials[material])
        if bl_prop is not None:
            for slot in bl_prop.texture_slots:
                export_animation.create_uv_controller(
                    bl_prop, slot, source_object, target_shape, ctx.options, ctx.cycle_type
                )
        return
    if bl_prop is None:
        create_fallback_property(target_shape, material)
    else:
        create_material_property(target_shape, bl_prop, ctx)
        create_alpha_property(target_shape, bl_prop)
        create_texturing_property(target_shape, bl_prop, source_object, ctx)
        create_wireframe_property(target_shape, source_object, bl_prop)
        create_stencil_property(target_shape, bl_prop)
    ctx.materials[material] = target_shape.properties


def create_material_property(target_shape, bl_prop, ctx: ExportMaterialContext) -> None:
    ni_prop = nif.NiMaterialProperty()
    ni_prop.name = bl_prop.material.name
    ni_prop.ambient_color[:] = bl_prop.ambient_color[:3]
    ni_prop.diffuse_color[:] = bl_prop.diffuse_color[:3]
    ni_prop.specular_color[:] = bl_prop.specular_color[:3]
    ni_prop.emissive_color[:] = bl_prop.emissive_color[:3]
    ni_prop.shine = bl_prop.shine
    ni_prop.alpha = bl_prop.alpha
    export_animation.create_material_controllers(ni_prop, bl_prop, ctx.options, ctx.cycle_type)
    target_shape.properties.append(ni_prop)


def create_alpha_property(target_shape, bl_prop) -> None:
    if bl_prop.needs_upgrade:
        raise nif_switch.SwitchError(
            f"Material '{bl_prop.material.name}' is from an older addon version; run Upgrade Morrowind Materials"
        )
    if not (bl_prop.use_alpha_blend or bl_prop.use_alpha_clip or bl_prop.no_sort):
        return
    ni_prop = nif.NiAlphaProperty(flags=bl_prop.alpha_flags)
    ni_prop.test_ref = int(bl_prop.alpha_threshold * 255)
    target_shape.properties.append(ni_prop)


def create_texturing_property(target_shape, bl_prop, source_object, ctx: ExportMaterialContext) -> None:
    if not any(slot.image for slot in bl_prop.texture_slots):
        return
    ni_prop = nif.NiTexturingProperty()
    for name in nif.NiTexturingProperty.texture_keys:
        create_texturing_property_map(target_shape, bl_prop, ni_prop, name, source_object, ctx)
    target_shape.properties.append(ni_prop)


def create_texturing_property_map(target_shape, bl_prop, ni_prop, name, source_object, ctx: ExportMaterialContext) -> None:
    try:
        bl_slot = getattr(bl_prop, name)
        filepath = bl_slot.image.filepath
    except AttributeError:
        return
    ni_slot = nif.NiTexturingPropertyMap()
    if not ctx.options.preserve_material_names:
        if (name == "base_texture") and filepath:
            prop = target_shape.get_property(nif.NiMaterialProperty)
            if prop is not None:
                prop.name = pathlib.Path(filepath).stem
    for i, uv in enumerate(source_object.data.uv_layers):
        if uv.name == bl_slot.layer:
            ni_slot.uv_set = i
            break
    if not bl_slot.use_repeat:
        ni_slot.clamp_mode = 0
    ni_slot.source = nif.NiSourceTexture()
    if not bl_slot.use_mipmaps:
        ni_slot.source.use_mipmaps = nif.NiSourceTexture.UseMipMaps.NO
    if (bl_slot.image.source == "GENERATED") or bl_slot.image.packed_file:
        include_alpha = bl_prop.use_alpha_blend or bl_prop.use_alpha_clip
        ni_slot.source.pixel_data = create_pixel_data(bl_slot.image, include_alpha, ctx.pixel_datas)
    else:
        ni_slot.source.filename = bpy.path.abspath(filepath)
    setattr(ni_prop, name, ni_slot)
    export_animation.create_uv_controller(bl_prop, bl_slot, source_object, target_shape, ctx.options, ctx.cycle_type)


def create_wireframe_property(target_shape, source_object, bl_prop) -> None:
    if source_object.display_type == "WIRE":
        target_shape.properties.append(nif.NiWireframeProperty(wireframe=True))


def create_stencil_property(target_shape, bl_prop) -> None:
    if not bl_prop.material.use_backface_culling:
        target_shape.properties.append(nif.NiStencilProperty(draw_mode=nif.NiStencilProperty.DrawMode.DRAW_BOTH))


def create_pixel_data(image, include_alpha, pixel_datas) -> nif.NiPixelData:
    if pixel_data := pixel_datas.get(image):
        assert pixel_data.pixel_format.has_alpha == include_alpha
    else:
        assert image.channels == 4
        width, height = image.size
        for size in (width, height):
            if (size == 0) or (size & (size - 1) != 0):
                raise nif_switch.SwitchError("Image dimensions must be a power of 2")
        if include_alpha:
            pixel_format = nif.NiPixelFormat.RGBA()
            pixel_stride = 4
        else:
            pixel_format = nif.NiPixelFormat.RGB()
            pixel_stride = 3
        pixel_data = nif.NiPixelData(pixel_format=pixel_format, pixel_stride=pixel_stride)
        pixel_data.mipmap_levels.resize((1, 3))
        pixel_data.mipmap_levels[:] = [width, height, 0]
        pixel_floats = np.empty((width, height, 4), np.float32)
        image.pixels.foreach_get(pixel_floats.ravel())
        if pixel_stride == 3:
            pixel_floats = pixel_floats[:, :, :3]
        pixel_floats = np.flipud(pixel_floats)
        pixel_data.pixel_data = (pixel_floats.ravel() * 255.0).round().astype(np.ubyte)
        pixel_datas[image] = pixel_data
    return pixel_data


def create_fallback_property(target_shape, material) -> None:
    bl_image = nif_shader.get_base_texture_image(material)
    if bl_image:
        ni_prop = nif.NiMaterialProperty()
        ni_prop.ambient_color[:] = 1.0
        ni_prop.diffuse_color[:] = 1.0
        ni_prop.alpha = 1.0
        target_shape.properties.append(ni_prop)
        ni_prop = nif.NiTexturingProperty()
        target_shape.properties.append(ni_prop)
        texture = ni_prop.base_texture = nif.NiTexturingPropertyMap()
        texture.source = nif.NiSourceTexture()
        texture.source.filename = bpy.path.abspath(bl_image.filepath)
