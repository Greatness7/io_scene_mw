from __future__ import annotations

from enum import IntEnum

from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiObject import NiObject


class BoundType(IntEnum):
    SPHERE_BV = 0
    BOX_BV = 1
    CAPSULE_BV = 2
    LOZENGE_BV = 3
    UNION_BV = 4
    HALFSPACE_BV = 5


class NiBoundingVolume(NiObject):  # TODO Not NiObject
    bound_type = BoundType.BOX_BV  # type: int32  # pyright: ignore[reportTypeCommentUsage]

    # provide access to related enums
    BoundType = BoundType

    @staticmethod
    def read(stream: NiBinaryStream) -> NiBoundingVolume:
        from .NiBoxBV import NiBoxBV
        from .NiSphereBV import NiSphereBV
        from .NiUnionBV import NiUnionBV

        bound_type = BoundType(stream.read_int())
        if bound_type == BoundType.SPHERE_BV:
            return stream.read_type(NiSphereBV)
        if bound_type == BoundType.BOX_BV:
            return stream.read_type(NiBoxBV)
        if bound_type == BoundType.UNION_BV:
            return stream.read_type(NiUnionBV)
        raise NotImplementedError(bound_type)

    def save(self, stream: NiBinaryStream) -> None:
        stream.write_int(self.bound_type)
