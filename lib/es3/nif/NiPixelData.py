from __future__ import annotations

from es3.utils.math import zeros
from .NiObject import NiObject
from .NiPixelFormat import NiPixelFormat


class NiPixelData(NiObject):
    pixel_format: NiPixelFormat = NiPixelFormat()
    palette: NiPalette | None = None
    pixel_stride: uint32 = 0
    mipmap_levels: ndarray = zeros(0, dtype="<I")
    pixel_data: ndarray = zeros(0, dtype="<B")

    _refs = (*NiObject._refs, "palette")

    def load(self, stream):
        self.pixel_format = stream.read_type(NiPixelFormat)
        self.palette = stream.read_link()
        mipmap_levels = stream.read_uint()
        self.pixel_stride = stream.read_uint()
        if mipmap_levels:
            self.mipmap_levels = stream.read_uints(mipmap_levels, 3)  # [width, height, offset]
        num_pixels = stream.read_uint()
        if num_pixels:
            self.pixel_data = stream.read_ubytes(num_pixels)

    def save(self, stream):
        self.pixel_format.save(stream)
        stream.write_link(self.palette)
        stream.write_uint(len(self.mipmap_levels))
        stream.write_uint(self.pixel_stride)
        stream.write_uints(self.mipmap_levels)
        stream.write_uint(len(self.pixel_data))
        stream.write_ubytes(self.pixel_data)


if __name__ == "__main__":
    from es3.nif import NiPalette
    from es3.utils.typing import *
