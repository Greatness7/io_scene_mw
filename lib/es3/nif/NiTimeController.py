from __future__ import annotations

from enum import IntEnum

from es3.utils.flags import bool_property, enum_property
from .NiObject import NiObject


class CycleType(IntEnum):
    CYCLE = 0
    REVERSE = 1
    CLAMP = 2


class NiTimeController(NiObject):
    next: NiTimeController | None = None
    flags: uint16 = 8
    frequency: float32 = 1.0
    phase: float32 = 0.0
    start_time: float32 = 0.0
    stop_time: float32 = 0.0
    target: NiObjectNET | None = None

    # provide access to related enums
    CycleType = CycleType

    # flags access
    cycle_type = enum_property(CycleType, mask=0x0006, pos=1)
    active = bool_property(mask=0x0008)

    _refs = (*NiObject._refs, "next")
    _ptrs = (*NiObject._ptrs, "target")

    def load(self, stream):
        self.next = stream.read_link()
        self.flags = stream.read_ushort()
        self.frequency = stream.read_float()
        self.phase = stream.read_float()
        self.start_time = stream.read_float()
        self.stop_time = stream.read_float()
        self.target = stream.read_link()

    def save(self, stream):
        stream.write_link(self.next)
        stream.write_ushort(self.flags)
        stream.write_float(self.frequency)
        stream.write_float(self.phase)
        stream.write_float(self.start_time)
        stream.write_float(self.stop_time)
        stream.write_link(self.target)

    def update_start_stop_times(self) -> tuple[int, int]:
        if self.data:
            self.start_time, self.stop_time = self.data.get_start_stop_times()
        else:
            self.start_time, self.stop_time = 0, 0


if __name__ == "__main__":
    from es3.nif import NiObjectNET
    from es3.utils.typing import *
