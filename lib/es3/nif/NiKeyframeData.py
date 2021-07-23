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

    def get_start_stop_times(self) -> tuple[int, int]:
        start_time = float('inf')
        stop_time = -start_time

        r = self.rotations.euler_data or (self.rotations,)
        t = self.translations
        s = self.scales

        for data in (*r, t, s):
            times = data.get_start_stop_times()
            start_time = min(start_time, times[0])
            stop_time = max(stop_time, times[1])

        return start_time, stop_time
