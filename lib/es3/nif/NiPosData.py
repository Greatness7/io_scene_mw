from __future__ import annotations

from .NiFloatData import KeyType, NiFloatData


class NiPosData(NiFloatData):

    @property
    def key_size(self):
        if self.interpolation == KeyType.LIN_KEY:
            return 4  # (time, x, y, z)
        if self.interpolation == KeyType.BEZ_KEY:
            return 10  # (time, x, y, z, inTan x, inTan y, inTan z, outTan x, outTan y, outTan z)
        if self.interpolation == KeyType.TCB_KEY:
            return 7  # (time, x, y, z, tension, continuity, bias)
        raise Exception(f"{self.type} does not support '{self.interpolation}'")

    @property
    def values(self):
        return self.keys[:, 1:4]

    @property
    def in_tans(self):
        return self.keys[:, 4:7]

    @property
    def out_tans(self):
        return self.keys[:, 7:10]

    @property
    def tcb(self):
        return self.keys[:, 4:7]

    def apply_scale(self, scale):
        if self.interpolation == KeyType.BEZ_KEY:
            self.keys[:, 1:] *= scale  # scale values and tangents
        else:
            self.keys[:, 1:4] *= scale  # scale values only
