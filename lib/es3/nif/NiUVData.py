from __future__ import annotations

from .NiFloatData import NiFloatData
from .NiObject import NiObject


class NiUVData(NiObject):
    u_offset_data: NiFloatData = NiFloatData()
    v_offset_data: NiFloatData = NiFloatData()
    u_tiling_data: NiFloatData = NiFloatData()
    v_tiling_data: NiFloatData = NiFloatData()

    def load(self, stream):
        self.u_offset_data = stream.read_type(NiFloatData)
        self.v_offset_data = stream.read_type(NiFloatData)
        self.u_tiling_data = stream.read_type(NiFloatData)
        self.v_tiling_data = stream.read_type(NiFloatData)

    def save(self, stream):
        self.u_offset_data.save(stream)
        self.v_offset_data.save(stream)
        self.u_tiling_data.save(stream)
        self.v_tiling_data.save(stream)

    def get_start_stop_times(self) -> tuple[int, int]:
        keys = list(filter(len, (
            self.u_offset_data.keys,
            self.v_offset_data.keys,
            self.u_tiling_data.keys,
            self.v_tiling_data.keys,
        )))
        if len(keys) == 0:
            return (0, 0)

        start_time = float('inf')
        stop_time = float('-inf')

        for item in keys:
            start_time = min(item[0, 0], start_time)
            stop_time = max(item[-1, 0], stop_time)

        return (start_time, stop_time)
