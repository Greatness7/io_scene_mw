from __future__ import annotations

from es3.utils.math import zeros
from .NiObject import NiObject


class NiPixelFormat(NiObject):
    pixel_format: uint32 = 0  # TODO enum
    color_masks: ndarray = zeros(4, dtype="<I")
    bits_per_pixel: uint32 = 0
    old_fast_compare: ndarray = zeros(8, dtype="<H")

    def load(self, stream):
        self.pixel_format = stream.read_uint()
        self.color_masks = stream.read_uints(4)
        self.bits_per_pixel = stream.read_uint()
        self.old_fast_compare = stream.read_ubytes(8)

    def save(self, stream):
        stream.write_uint(self.pixel_format)
        stream.write_uints(self.color_masks)
        stream.write_uint(self.bits_per_pixel)
        stream.write_ubytes(self.old_fast_compare)


if __name__ == "__main__":
    from es3.utils.typing import *
