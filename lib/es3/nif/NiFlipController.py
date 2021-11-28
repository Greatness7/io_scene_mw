from __future__ import annotations

from enum import IntEnum

from .NiTimeController import NiTimeController


class AffectedMap(IntEnum):
    BASE_MAP = 0
    DARK_MAP = 1
    DETAIL_MAP = 2
    GLOSS_MAP = 3
    GLOW_MAP = 4
    BUMP_MAP = 5
    DECAL_0_MAP = 6
    DECAL_1_MAP = 7
    DECAL_2_MAP = 8
    DECAL_3_MAP = 9


class NiFlipController(NiTimeController):
    affected_map: int32 = AffectedMap.BASE_MAP
    flip_start_time: float32 = 0.0
    secs_per_frame: float32 = 0.0
    textures: list[NiSourceTexture] = []

    # provide access to related enums
    AffectedMap = AffectedMap

    _refs = (*NiTimeController._refs, "textures")

    def load(self, stream):
        super().load(stream)
        self.affected_map = stream.read_int()
        self.flip_start_time = stream.read_float()
        self.secs_per_frame = stream.read_float()
        self.textures = stream.read_links()

    def save(self, stream):
        super().save(stream)
        stream.write_int(self.affected_map)
        stream.write_float(self.flip_start_time)
        stream.write_float(self.secs_per_frame)
        stream.write_links(self.textures)


if __name__ == "__main__":
    from es3.nif import NiSourceTexture
    from es3.utils.typing import *
