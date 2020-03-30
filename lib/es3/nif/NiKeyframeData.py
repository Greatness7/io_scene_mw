from __future__ import annotations

from .NiFloatData import NiFloatData
from .NiObject import NiObject
from .NiPosData import NiPosData
from .NiRotData import NiRotData


class NiKeyframeData(NiObject):
    rotations: NiRotData = NiRotData()
    translations: NiPosData = NiPosData()
    scales: NiFloatData = NiFloatData()

    def load(self, stream):
        self.rotations = stream.read_type(NiRotData)
        self.translations = stream.read_type(NiPosData)
        self.scales = stream.read_type(NiFloatData)

    def save(self, stream):
        self.rotations.save(stream)
        self.translations.save(stream)
        self.scales.save(stream)

    def apply_scale(self, scale):
        self.translations.apply_scale(scale)
