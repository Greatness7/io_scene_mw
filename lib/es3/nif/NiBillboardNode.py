from __future__ import annotations

from enum import IntEnum

from es3.utils.flags import enum_property
from .NiNode import NiNode


class BillboardMode(IntEnum):
    ALWAYS_FACE_CAMERA = 0
    ROTATE_ABOUT_UP = 1
    RIGID_FACE_CAMERA = 2
    ALWAYS_FACE_CENTER = 3


class NiBillboardNode(NiNode):

    # provide access to related enums
    BillboardMode = BillboardMode

    # flags access
    billboard_mode = enum_property(BillboardMode, mask=0x0060, pos=5)

