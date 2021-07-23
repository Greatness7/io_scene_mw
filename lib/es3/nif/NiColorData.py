from __future__ import annotations

from .NiFloatData import KeyType, NiFloatData


class NiColorData(NiFloatData):

    @property
    def key_size(self) -> int:
        if self.key_type == KeyType.LIN_KEY:
            return 5  # (time, r, g, b, a)
        raise Exception(f"{self.type} does not support '{self.key_type}'")
