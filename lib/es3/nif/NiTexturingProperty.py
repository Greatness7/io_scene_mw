from __future__ import annotations

from enum import IntEnum
from operator import attrgetter

from .NiProperty import NiProperty
from .NiTexturingPropertyBumpMap import NiTexturingPropertyBumpMap
from .NiTexturingPropertyMap import NiTexturingPropertyMap


class ApplyMode(IntEnum):
    APPLY_REPLACE = 0
    APPLY_DECAL = 1
    APPLY_MODULATE = 2
    APPLY_HILIGHT = 3
    APPLY_HILIGHT2 = 4


class TextureMaps(IntEnum):
    BASE_TEXTURE = 0
    DARK_TEXTURE = 1
    DETAIL_TEXTURE = 2
    GLOSS_TEXTURE = 3
    GLOW_TEXTURE = 4
    BUMP_MAP_TEXTURE = 5
    DECAL_0_TEXTURE = 6
    DECAL_1_TEXTURE = 7
    DECAL_2_TEXTURE = 8
    DECAL_3_TEXTURE = 9


class _TextureMapSource:
    __slots__ = "name",

    def __set_name__(self, owner, name):
        # e.g. "_base_texture_source" -> "base_texture"
        self.name = name[1:-7]

    def __get__(self, instance, owner):
        texture_map = getattr(instance, self.name, None)
        return texture_map and texture_map.source

    def __set__(self, instance, value):
        texture_map = getattr(instance, self.name, None)
        texture_map.source = value


class NiTexturingProperty(NiProperty):
    apply_mode: int32 = ApplyMode.APPLY_MODULATE
    base_texture: NiTexturingPropertyMap | None = None
    dark_texture: NiTexturingPropertyMap | None = None
    detail_texture: NiTexturingPropertyMap | None = None
    gloss_texture: NiTexturingPropertyMap | None = None
    glow_texture: NiTexturingPropertyMap | None = None
    bump_map_texture: NiTexturingPropertyBumpMap | None = None
    decal_0_texture: NiTexturingPropertyMap | None = None
    decal_1_texture: NiTexturingPropertyMap | None = None
    decal_2_texture: NiTexturingPropertyMap | None = None
    decal_3_texture: NiTexturingPropertyMap | None = None

    # provide access to related enums
    ApplyMode = ApplyMode

    # convenience properties
    texture_keys = tuple(e.name.lower() for e in TextureMaps)
    texture_maps = property(attrgetter(*texture_keys))

    _refs = (*NiProperty._refs, *(f"_{k}_source" for k in texture_keys))

    # internal private members
    _base_texture_source = _TextureMapSource()
    _dark_texture_source = _TextureMapSource()
    _detail_texture_source = _TextureMapSource()
    _gloss_texture_source = _TextureMapSource()
    _glow_texture_source = _TextureMapSource()
    _bump_map_texture_source = _TextureMapSource()
    _decal_0_texture_source = _TextureMapSource()
    _decal_1_texture_source = _TextureMapSource()
    _decal_2_texture_source = _TextureMapSource()
    _decal_3_texture_source = _TextureMapSource()

    def load(self, stream):
        super().load(stream)
        self.apply_mode = ApplyMode(stream.read_int())
        num_texture_maps = stream.read_uint()
        for i in range(num_texture_maps):
            has_texture_map = stream.read_bool()
            if has_texture_map:
                item = stream.read_type(NiTexturingPropertyBumpMap if i == 5 else NiTexturingPropertyMap)
                setattr(self, self.texture_keys[i], item)

    def save(self, stream):
        super().save(stream)
        stream.write_int(self.apply_mode)
        texture_maps = self._get_active_texture_maps()
        stream.write_uint(len(texture_maps))
        for item in texture_maps:
            stream.write_bool(item)
            if item:
                item.save(stream)

    def _get_active_texture_maps(self):
        maps = self.texture_maps
        required, optional = maps[:7], maps[7:]
        return (*required, *filter(None, optional))


if __name__ == "__main__":
    from es3.utils.typing import *
