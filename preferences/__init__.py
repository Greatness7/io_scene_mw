import bpy

from .addon_prefs import Preferences, TexturePath
from .texture_path import (
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
