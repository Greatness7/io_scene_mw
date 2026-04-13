from __future__ import annotations

from ..utils.math import zeros
from ..utils.typing import *

from .NiObject import NiObject


class NiPalette(NiObject):
    has_alpha: uint8 = 0
    palettes: ndarray = zeros(0, 4, dtype="<B")

    def load(self, stream):
        self.has_alpha = stream.read_ubyte()
        num_palettes = stream.read_uint()
        if num_palettes:
            self.palettes = stream.read_ubytes(num_palettes, 4)

    def save(self, stream):
        stream.write_ubyte(self.has_alpha)
        stream.write_uint(len(self.palettes))
        stream.write_ubytes(self.palettes)
