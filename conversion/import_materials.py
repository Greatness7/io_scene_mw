"""Create or reuse Blender materials for imported NIF shapes.

Materials use the addon's Morrowind shader. The import context holds the material cache shared across shapes
in one import and supplies the texture search paths.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, cast

import bpy
from es3 import nif

from .. import nif_shader
from . import import_animation
from .import_animation import ImportAnimationContext
from .options import ImportOptions


@dataclass(frozen=True, slots=True)
class ImportMaterialContext:
    options: ImportOptions
    texture_paths: tuple[pathlib.Path, ...]
    materials: dict[tuple[Any, ...], Any]
    animation: ImportAnimationContext


def import_material(source_shape, properties, target_object, ctx: ImportMaterialContext) -> None:
    if len(properties) == 0:
        return
    ni_alpha = properties.get(nif.NiAlphaProperty)
    ni_material = properties.get(nif.NiMaterialProperty)
    ni_stencil = properties.get(nif.NiStencilProperty)
    ni_texture = properties.get(nif.NiTexturingProperty)
    ni_wireframe = properties.get(nif.NiWireframeProperty)
    if (ni_texture is None) and len(source_shape.data.vertex_colors):
        ni_texture = nif.NiTexturingProperty()
    if ni_wireframe and ni_wireframe.wireframe:
        target_object.display_type = "WIRE"
    name = calc_name_from_textures(ni_texture, ctx.options)
    if apply_existing_material(name, ni_alpha, source_shape, target_object, ctx.options):
        return
    props_hash = (
        *properties.values(),
        len(source_shape.data.vertex_colors),
        source_shape.controllers.find_type(nif.NiUVController),
    )
    bl_prop = None
    try:
        bl_prop = ctx.materials[props_hash]
    except KeyError:
        bl_prop = ctx.materials[props_hash] = nif_shader.execute(target_object)
    else:
        link_object_material(target_object, bl_prop.material)
        return
    finally:
        if ctx.options.use_existing_materials and bl_prop is not None:
            bl_prop.material.name = name
    assert bl_prop is not None
    if ni_alpha:
        create_alpha_property(bl_prop, ni_alpha)
    if ni_material:
        create_material_property(bl_prop, ni_material, ctx)
    if ni_stencil:
        create_stencil_property(bl_prop, ni_stencil)
    if ni_texture:
        create_texturing_property(bl_prop, ni_texture, source_shape, target_object, ctx)
    if ni_wireframe:
        create_wireframe_property(target_object, ni_wireframe)


def create_alpha_property(bl_prop, ni_prop) -> None:
    bl_prop.alpha_flags = ni_prop.flags
    bl_prop.alpha_threshold = float(ni_prop.test_ref / 255)
    bl_prop.sync_alpha_state()


def create_material_property(bl_prop, ni_prop, ctx: ImportMaterialContext) -> None:
    if not ctx.options.use_existing_materials:
        bl_prop.material.name = ni_prop.name
    bl_prop.ambient_color[:3] = ni_prop.ambient_color
    bl_prop.diffuse_color[:3] = ni_prop.diffuse_color
    bl_prop.specular_color[:3] = ni_prop.specular_color
    bl_prop.emissive_color[:3] = ni_prop.emissive_color
    bl_prop.shine = ni_prop.shine
    bl_prop.alpha = ni_prop.alpha
    import_animation.create_color_controller(bl_prop, ni_prop, ctx.animation)
    import_animation.create_alpha_controller(bl_prop, ni_prop, ctx.animation)


def create_texturing_property(bl_prop, ni_prop, source_shape, target_object, ctx: ImportMaterialContext) -> None:
    for name in nif.NiTexturingProperty.texture_keys:
        create_texturing_property_map(bl_prop, ni_prop, name, target_object, ctx.texture_paths)
    if target_object.data.vertex_colors:
        bl_prop.vertex_color.layer_name = target_object.data.vertex_colors[0].name
        bl_prop.create_link(bl_prop.vertex_color, bl_prop.shader, "Color", "Vertex Color")
        bl_prop.create_link(bl_prop.vertex_color, bl_prop.shader, "Alpha", "Diffuse Alpha")
        bl_prop.inputs["Use Vertex Colors"].default_value = 1.0
    for controller in source_shape.controllers:
        if isinstance(controller, nif.NiUVController):
            import_animation.create_uv_controller(controller, target_object, ctx.animation)


def create_wireframe_property(target_object, ni_prop) -> None:
    if ni_prop.wireframe:
        target_object.display_type = "WIRE"


def create_stencil_property(bl_prop, ni_prop) -> None:
    bl_prop.material.use_backface_culling = False
    bl_prop.material.show_transparent_back = True


def create_texturing_property_map(bl_prop, ni_prop, slot_name, target_object, texture_paths) -> None:
    try:
        bl_slot = getattr(bl_prop, slot_name)
        ni_slot = getattr(ni_prop, slot_name)
        image = create_image(ni_slot.source.filename, texture_paths)
    except (AttributeError, LookupError):
        return
    bl_slot.image = image
    if ni_slot.clamp_mode.name == "CLAMP_S_CLAMP_T":
        bl_slot.use_repeat = False
    if ni_slot.source.use_mipmaps.name == "NO":
        bl_slot.use_mipmaps = False
    try:
        bl_slot.layer = target_object.data.uv_layers[ni_slot.uv_set].name
    except IndexError:
        pass


def create_image(filepath, texture_paths):
    abspath = resolve_texture_path(filepath, texture_paths)
    if abspath.exists():
        return bpy.data.images.load(str(abspath), check_existing=True)
    image = bpy.data.images.new(name=abspath.name, width=1, height=1)
    image.filepath = str(abspath)
    image.source = "FILE"
    return image


def calc_name_from_textures(ni_prop, options: ImportOptions) -> str:
    if not options.use_existing_materials or ni_prop is None:
        return ""
    names = {}
    for tex_key, tex_map in zip(ni_prop.texture_keys, ni_prop.texture_maps):
        try:
            names[tex_key] = pathlib.Path(tex_map.source.filename).stem.lower()
        except AttributeError:
            pass
    if names.keys() == {"base_texture"}:
        return names["base_texture"]
    return " | ".join(f"{key.rpartition('_')[0]}:{value}" for key, value in names.items())


def apply_existing_material(name, ni_alpha, source_shape, target_object, options: ImportOptions):
    if not options.use_existing_materials:
        return
    use_vertex_colors = bool(len(source_shape.data.vertex_colors))
    base_name, index = name, 0
    while True:
        try:
            material = cast(Any, bpy.data.materials[name])
            bl_prop = material.mw.validate()
        except (LookupError, TypeError):
            break
        if bl_prop.use_vertex_colors == use_vertex_colors and alpha_matches(bl_prop, ni_alpha):
            link_object_material(target_object, bl_prop.material)
            return True
        index += 1
        name = f"{base_name}.{index:03}"


def alpha_matches(bl_prop, ni_alpha) -> bool:
    """Whether the material exports `ni_alpha`: the same flags and threshold, or no alpha property at all."""
    if not (bl_prop.use_alpha_blend or bl_prop.use_alpha_clip or bl_prop.no_sort):
        return ni_alpha is None
    return (
        ni_alpha is not None
        and bl_prop.alpha_flags == ni_alpha.flags
        and int(bl_prop.alpha_threshold * 255) == ni_alpha.test_ref
    )


def link_object_material(bl_object, bl_material) -> None:
    for slot in bl_object.material_slots:
        if slot.material is None:
            break
    else:
        bl_object.data.materials.append(None)
        slot = bl_object.material_slots[-1]
    slot.link = "OBJECT"
    slot.material = bl_material


def resolve_texture_path(relpath, texture_paths, case_insensitive=pathlib.Path(__file__.upper()).exists()):
    path = pathlib.Path(bpy.path.native_pathsep(relpath).lower())
    if path.parts[0] == "data files":
        path = path.relative_to("data files")
    if path.parts[0] == "textures":
        path = path.relative_to("textures")
    suffixes = {path.suffix, ".dds", ".tga", ".bmp"}
    for directory in texture_paths:
        abspath = directory / path
        for suffix in suffixes:
            abspath = abspath.with_suffix(suffix)
            if not case_insensitive:
                abspath = pathlib.Path(bpy.path.resolve_ncase(str(abspath)))
            if abspath.exists():
                return abspath
    return pathlib.Path("textures") / path
