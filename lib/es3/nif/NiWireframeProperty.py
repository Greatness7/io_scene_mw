from __future__ import annotations

from es3.utils.flags import bool_property
from .NiProperty import NiProperty


class NiWireframeProperty(NiProperty):
    wireframe = bool_property(mask=0x0001)
