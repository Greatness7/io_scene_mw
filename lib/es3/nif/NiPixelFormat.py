from __future__ import annotations

from enum import IntEnum

from es3.utils.math import zeros
from .NiObject import NiObject


class PixelFormat(IntEnum):
    RGB = 0
    RGBA = 1
    PAL = 2
    PALALPHA = 3
    BGR = 4
    BGRA = 5
    COMPRESS1 = 6
    COMPRESS3 = 7
    COMPRESS5 = 8
    RGB24NONINTERLEAVED = 9
    BUMP = 10
    BUMPLUMA = 11


class NiPixelFormat(NiObject):
    format: int32 = PixelFormat.RGB
    color_masks: ndarray = zeros(4, dtype="<I")
    bits_per_pixel: uint32 = 0
    compare_bits: ndarray = zeros(2, dtype="<I")

    # provide access to related enums
    PixelFormat = PixelFormat

    def load(self, stream):
        self.format = PixelFormat(stream.read_int())
        self.color_masks = stream.read_uints(4)
        self.bits_per_pixel = stream.read_uint()
        self.compare_bits = stream.read_uints(2)

    def save(self, stream):
        stream.write_int(self.format)
        stream.write_uints(self.color_masks)
        stream.write_uint(self.bits_per_pixel)
        stream.write_uints(self.compare_bits)

    @property
    def has_alpha(self):
        return self.format in (
            PixelFormat.RGBA,
            PixelFormat.PALALPHA,
            PixelFormat.BGRA,
            PixelFormat.COMPRESS3,
            PixelFormat.COMPRESS5,
        )

    @classmethod
    @property
    def RGB(cls) -> NiPixelFormat:
        pixel_format = cls()
        pixel_format.format = PixelFormat.RGB
        pixel_format.color_masks[:] = [0x000000FF, 0x0000FF00, 0x00FF0000, 0x00000000]
        pixel_format.bits_per_pixel = 24
        pixel_format.compare_bits[:] = [0x00820860, 0x00004100]
        return pixel_format

    @classmethod
    @property
    def RGBA(cls) -> NiPixelFormat:
        pixel_format = cls()
        pixel_format.format = PixelFormat.RGBA
        pixel_format.color_masks[:] = [0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000]
        pixel_format.bits_per_pixel = 32
        pixel_format.compare_bits[:] = [0x20820881, 0x000C4100]
        return pixel_format

    @classmethod
    @property
    def DXT1(cls) -> NiPixelFormat:
        pixel_format = cls()
        pixel_format.format = PixelFormat.COMPRESS1
        pixel_format.color_masks[:] = [0x00000000, 0x00000000, 0x00000000, 0x00000000]
        pixel_format.bits_per_pixel = 0
        pixel_format.compare_bits[:] = [0x00000006, 0x00000000]
        return pixel_format

    @classmethod
    @property
    def DXT5(cls) -> NiPixelFormat:
        pixel_format = cls()
        pixel_format.format = PixelFormat.COMPRESS5
        pixel_format.color_masks[:] = [0x00000000, 0x00000000, 0x00000000, 0x00000000]
        pixel_format.bits_per_pixel = 0
        pixel_format.compare_bits[:] = [0x00000008, 0x00000000]
        return pixel_format


if __name__ == "__main__":
    from es3.utils.typing import *
