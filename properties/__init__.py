import bpy

from .material_properties import MaterialProperties, TextureProperties
from .object_properties import ObjectProperties

classes = (
    TextureProperties,
    MaterialProperties,
    ObjectProperties,
)

register, unregister = bpy.utils.register_classes_factory(classes)
