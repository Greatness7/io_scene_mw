from __future__ import annotations

from ..utils.flags import bool_property
from ..utils.typing import *

from .NiSwitchNode import NiSwitchNode


class NiFltAnimationNode(NiSwitchNode):
    period: float32 = 0

    # flags access
    bounce = bool_property(mask=0x0040)

    def load(self, stream):
        super().load(stream)
        self.period = stream.read_float()

    def save(self, stream):
        super().save(stream)
        stream.write_float(self.period)

    def apply_time_scale(self, scale: float):
        super().apply_time_scale(scale)
        self.period *= scale
