from __future__ import annotations

from enum import IntEnum

from .NiTimeController import NiTimeController


class BankDirection(IntEnum):
    NEGATIVE = -1
    POSITIVE = 1


class FollowAxis(IntEnum):
    AXIS_X = 0
    AXIS_Y = 1
    AXIS_Z = 2


class NiPathController(NiTimeController):
    bank_direction: int32 = BankDirection.POSITIVE
    max_bank_angle: float32 = 0.0
    smoothing: float32 = 0.0
    follow_axis: int16 = FollowAxis.AXIS_X
    path_data: NiPosData | None = None
    percentage_data: NiFloatData | None = None

    # provide access to related enums
    BankDirection = BankDirection
    FollowAxis = FollowAxis

    _refs = (*NiTimeController._refs, "path_data", "percentage_data")

    def load(self, stream):
        super().load(stream)
        self.bank_direction = BankDirection(stream.read_int())
        self.max_bank_angle = stream.read_float()
        self.smoothing = stream.read_float()
        self.follow_axis = FollowAxis(stream.read_short())
        self.path_data = stream.read_link()
        self.percentage_data = stream.read_link()

    def save(self, stream):
        super().save(stream)
        stream.write_int(self.bank_direction)
        stream.write_float(self.max_bank_angle)
        stream.write_float(self.smoothing)
        stream.write_short(self.follow_axis)
        stream.write_link(self.path_data)
        stream.write_link(self.percentage_data)


if __name__ == "__main__":
    from es3.nif import NiFloatData, NiPosData
    from es3.utils.typing import *
