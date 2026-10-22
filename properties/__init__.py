import bpy

from .material_properties import MaterialProperties, TextureProperties
from .object_properties import LODRange, ObjectProperties

classes = (
    TextureProperties,
    MaterialProperties,
    LODRange,
    ObjectProperties,
)

register, unregister = bpy.utils.register_classes_factory(classes)
