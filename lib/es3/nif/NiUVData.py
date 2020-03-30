from __future__ import annotations

from .NiFloatData import NiFloatData
from .NiObject import NiObject


class NiUVData(NiObject):
    offset_u: NiFloatData = NiFloatData()
    offset_v: NiFloatData = NiFloatData()
    tiling_u: NiFloatData = NiFloatData()
    tiling_v: NiFloatData = NiFloatData()

    def load(self, stream):
        self.offset_u = stream.read_type(NiFloatData)
        self.offset_v = stream.read_type(NiFloatData)
        self.tiling_u = stream.read_type(NiFloatData)
        self.tiling_v = stream.read_type(NiFloatData)

    def save(self, stream):
        self.offset_u.save(stream)
        self.offset_v.save(stream)
        self.tiling_u.save(stream)
        self.tiling_v.save(stream)

    def get_start_stop_times(self):
        keys = list(filter(len, (
            self.offset_u.keys,
            self.offset_v.keys,
            self.tiling_u.keys,
            self.tiling_v.keys,
        )))
        if len(keys) == 0:
            return (0, 0)

        start_time = float('inf')
        stop_time = float('-inf')

        for item in keys:
            start_time = min(item[0, 0], start_time)
            stop_time = max(item[-1, 0], stop_time)

        return (start_time, stop_time)
