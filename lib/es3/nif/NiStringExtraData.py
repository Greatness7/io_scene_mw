from __future__ import annotations

from .NiBinaryStream import NiBinaryStream
from .NiExtraData import NiExtraData


class NiStringExtraData(NiExtraData):
    string_data: str = ""

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        self.string_data = stream.read_str()

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_str(self.string_data)
