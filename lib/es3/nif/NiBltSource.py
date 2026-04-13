from __future__ import annotations

from ..utils.typing import *

from .NiObject import NiObject
from .NiPixelData import NiPixelData


class NiBltSource(NiObject):
    filename: str = ""
    pixel_data: NiPixelData | None = None

    def load(self, stream):
        has_external_texture = stream.read_ubyte()
        if has_external_texture:
            self.filename = stream.read_str()
        else:
            has_pixel_data = stream.read_ubyte()
            if has_pixel_data:
                self.pixel_data = stream.read_link()

    def save(self, stream):
        stream.write_ubyte(bool(self.filename))
        if self.filename:
            stream.write_str(self.filename)
        else:
            stream.write_ubyte(bool(self.pixel_data))
            if self.pixel_data:
                self.pixel_data.save(stream)
