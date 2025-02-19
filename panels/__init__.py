import bpy

from .markers_panel import (
    MarkersList,
    MarkersListCopy,
    MarkersListMenu,
    MarkersListPaste,
    MarkersListSort,
    MarkersPanel,
)
from .material_panel import (
    BaseTexturePanel,
    DarkTexturePanel,
    Decal0TexturePanel,
    Decal1TexturePanel,
    Decal2TexturePanel,
    Decal3TexturePanel,
    DetailTexturePanel,
    GlowTexturePanel,
    MaterialPanel,
    MaterialPanelSettings,
)
from .object_panel import ObjectPanel

classes = (
    MarkersListCopy,
    MarkersListPaste,
    MarkersListSort,
    MarkersListMenu,
    MarkersList,
    MarkersPanel,
    MaterialPanel,
    MaterialPanelSettings,
    BaseTexturePanel,
    DarkTexturePanel,
    DetailTexturePanel,
    GlowTexturePanel,
    Decal0TexturePanel,
    Decal1TexturePanel,
    Decal2TexturePanel,
    Decal3TexturePanel,
    ObjectPanel,
)

register, unregister = bpy.utils.register_classes_factory(classes)
