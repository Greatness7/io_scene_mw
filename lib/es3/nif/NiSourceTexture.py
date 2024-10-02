from __future__ import annotations

import pathlib
from enum import IntEnum

from .NiTexture import NiTexture


class PixelLayout(IntEnum):
    PALETTIZED_8 = 0
    HIGH_COLOR_16 = 1
    TRUE_COLOR_32 = 2
    COMPRESSED = 3
    BUMPMAP = 4
    PALETTIZED_4 = 5
    PIX_DEFAULT = 6


class UseMipMaps(IntEnum):
    NO = 0
    YES = 1
    MIP_DEFAULT = 2


class AlphaFormat(IntEnum):
    NONE = 0
    BINARY = 1
    SMOOTH = 2
    ALPHA_DEFAULT = 3


class NiSourceTexture(NiTexture):
    filename: str = ""
    pixel_data: NiPixelData | None = None
    pixel_layout: int32 = PixelLayout.PALETTIZED_4
    use_mipmaps: int32 = UseMipMaps.YES
    alpha_format: int32 = AlphaFormat.ALPHA_DEFAULT
    is_static: uint8 = 1

    # provide access to related enums
    PixelLayout = PixelLayout
    UseMipMaps = UseMipMaps
    AlphaFormat = AlphaFormat

    _refs = (*NiTexture._refs, "pixel_data")

    def load(self, stream):
        super().load(stream)
        has_external_texture = stream.read_ubyte()
        if has_external_texture:
            self.filename = stream.read_str()
        else:
            has_pixel_data = stream.read_ubyte()
            if has_pixel_data:
                self.pixel_data = stream.read_link()
        self.pixel_layout = PixelLayout(stream.read_int())
        self.use_mipmaps = UseMipMaps(stream.read_int())
        self.alpha_format = AlphaFormat(stream.read_int())
        self.is_static = stream.read_ubyte()

    def save(self, stream):
        super().save(stream)
        stream.write_ubyte(bool(self.filename))
        if self.filename:
            stream.write_str(self.filename)
        else:
            stream.write_ubyte(bool(self.pixel_data))
            if self.pixel_data:
                stream.write_link(self.pixel_data)
        stream.write_int(self.pixel_layout)
        stream.write_int(self.use_mipmaps)
        stream.write_int(self.alpha_format)
        stream.write_ubyte(self.is_static)

    def sanitize_filename(self):
        filename = self.filename.lower().replace("\\", pathlib.os.sep)

        try:
            path = pathlib.Path(filename)
        except ValueError:
            path = pathlib.Path()

        # temporary file name so we can return early
        self.filename = path.name

        if not path.suffix:
            return
        if len(path.parts) <= 1:
            return
        if len(path.parts) == 2:
            # if there is only one parent folder and
            # that folder is textures, then omitting
            # everything except file name is allowed
            if path.parent.name == "textures":
                return

        try:
            textures_index = path.parts[::-1].index("textures")
        except ValueError:
            path = "textures" / path  # did not find textures dir
        else:
            if textures_index == 1:  # textures is the parent dir
                return
            path = path.relative_to(path.parents[textures_index])

        self.filename = str(path).replace(pathlib.os.sep, "\\")


if __name__ == "__main__":
    from es3.nif import NiPixelData
    from es3.utils.typing import *
