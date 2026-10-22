from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..utils.flags import bool_property
from ..utils.typing import *

from .NiAVObject import NiAVObject
from .NiBinaryStream import NiBinaryStream
from .NiNode import NiNode


class NiSwitchNode(NiNode):
    active_index: int32 = 0  # signed: -1 selects no child

    # flags access
    update_only_active = bool_property(mask=0x0020)

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        self.active_index = stream.read_int()

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_int(self.active_index)

    def sort(self, key: Callable[[Any], Any] = lambda value: value) -> None:
        """Canonicalize non-positional fields without reordering children.

        The key is accepted for signature compatibility and intentionally ignored.
        """
        NiAVObject.sort(self)
