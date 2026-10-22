from __future__ import annotations

import numpy as np

from ..utils.math import ZERO3, zeros
from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiSwitchNode import NiSwitchNode

NO_LIMIT = np.finfo(np.float32).max  # a far of float32 max means "no limit"


class NiLODNode(NiSwitchNode):
    lod_center: NiPoint3 = ZERO3
    lod_levels: ndarray = zeros(0, 2)

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        self.lod_center = stream.read_floats(3)
        num_lod_levels = stream.read_uint()
        if num_lod_levels:
            self.lod_levels = stream.read_floats(num_lod_levels, 2)

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_floats(self.lod_center)
        stream.write_uint(len(self.lod_levels))
        stream.write_floats(self.lod_levels)

    def apply_scale(self, scale: float) -> None:
        super().apply_scale(scale)
        self.lod_center = self.lod_center * scale  # not in place: the default is a shared read-only array
        levels = zeros(*self.lod_levels.shape)
        levels[:] = self.lod_levels
        finite = levels < NO_LIMIT  # scaling the no-limit sentinel would overflow to inf
        levels[finite] *= scale
        self.lod_levels = levels
