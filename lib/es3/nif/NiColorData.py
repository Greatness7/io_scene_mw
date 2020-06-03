from __future__ import annotations

from .NiFloatData import KeyType, NiFloatData


class NiColorData(NiFloatData):

    @property
    def key_size(self):
        if self.interpolation == KeyType.NO_INTERP:
            return 5  # (time, r, g, b, a)
        if self.interpolation == KeyType.LIN_KEY:
            return 5  # (time, r, g, b, a)
        raise Exception(f"{self.type} does not support '{self.interpolation}'")
