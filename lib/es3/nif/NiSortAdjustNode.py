from __future__ import annotations

from enum import IntEnum

from .NiNode import NiNode


class SortingMode(IntEnum):
    SORTING_INHERIT = 0
    SORTING_OFF = 1
    SORTING_SUBSORT = 2


class NiSortAdjustNode(NiNode):
    sorting_mode: int32 = SortingMode.SORTING_OFF
    sub_sorter: NiAccumulator | None = None

    # provide access to related enums
    SortingMode = SortingMode

    _refs = (*NiNode._refs, "sub_sorter")

    def load(self, stream):
        super().load(stream)
        self.sorting_mode = SortingMode(stream.read_int())
        self.sub_sorter = stream.read_link()

    def save(self, stream):
        super().save(stream)
        stream.write_int(self.sorting_mode)
        stream.write_link(self.sub_sorter)


if __name__ == "__main__":
    from es3.nif import NiAccumulator
    from es3.utils.typing import *
