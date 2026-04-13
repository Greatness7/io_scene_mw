from __future__ import annotations

from ..utils.math import zeros
from ..utils.typing import *

from .NiAVObject import NiAVObject


class NiDynamicEffect(NiAVObject):
    affected_nodes: ndarray = zeros(0, dtype="<i")

    def load(self, stream):
        super().load(stream)
        num_affected_nodes = stream.read_uint()
        if num_affected_nodes:
            self.affected_nodes = stream.read_ints(num_affected_nodes)

    def save(self, stream):
        super().save(stream)
        stream.write_uint(len(self.affected_nodes))
        stream.write_ints(self.affected_nodes)
