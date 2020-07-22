from __future__ import annotations

from enum import IntEnum

from .NiObject import NiObject


class BoundType(IntEnum):
    SPHERE_BV = 0
    BOX_BV = 1
    CAPSULE_BV = 2
    LOZENGE_BV = 3
    UNION_BV = 4
    HALFSPACE_BV = 5


class NiBoundingVolume(NiObject):  # TODO Not NiObject
    bound_type = BoundType.BOX_BV  # type: int32

    # provide access to related enums
    BoundType = BoundType

    @classmethod
    def load(cls, stream):
        if cls is not NiBoundingVolume:
            return  # ignore subclasses
        bound_type = BoundType(stream.read_int())
        if bound_type == BoundType.SPHERE_BV:
            return stream.read_type(NiSphereBV)
        if bound_type == BoundType.BOX_BV:
            return stream.read_type(NiBoxBV)
        if bound_type == BoundType.UNION_BV:
            return stream.read_type(NiUnionBV)
        raise NotImplementedError(bound_type)

    @classmethod
    def save(cls, stream):
        stream.write_int(cls.bound_type)


from .NiSphereBV import NiSphereBV
from .NiBoxBV import NiBoxBV
from .NiUnionBV import NiUnionBV


if __name__ == "__main__":
    from es3.utils.typing import *
