from __future__ import annotations

from .NiExtraData import NiExtraData


class NiStringExtraData(NiExtraData):
    string_data: str = ""

    def load(self, stream):
        super().load(stream)
        self.string_data = stream.read_str()

    def save(self, stream):
        super().save(stream)
        stream.write_str(self.string_data)
