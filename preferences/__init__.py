import bpy

from .addon_prefs import Preferences
from .texture_path import (
    TexturePath,
    TexturePathAdd,
    TexturePathList,
    TexturePathMove,
    TexturePathRemove,
)

classes = (
    TexturePathList,
    TexturePath,
    TexturePathAdd,
    TexturePathRemove,
    TexturePathMove,
    Preferences,
)

register, unregister = bpy.utils.register_classes_factory(classes)
