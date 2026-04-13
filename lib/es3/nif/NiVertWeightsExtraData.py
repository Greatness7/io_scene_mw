from __future__ import annotations

from ..utils.math import zeros
from ..utils.typing import *

from .NiExtraData import NiExtraData


class NiVertWeightsExtraData(NiExtraData):
    weights: ndarray = zeros(0)

    def load(self, stream):
        super().load(stream)
        num_weights = stream.read_ushort()
        if num_weights:
            self.weights = stream.read_floats(num_weights)

    def save(self, stream):
        super().save(stream)
        stream.write_ushort(len(self.weights))
        stream.write_floats(self.weights)
