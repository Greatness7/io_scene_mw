from __future__ import annotations

from es3.utils.flags import bool_property
from .NiNode import NiNode


class NiBSAnimationNode(NiNode):

    # flags access
    animated = bool_property(mask=0x0020)
    not_random = bool_property(mask=0x0040)
