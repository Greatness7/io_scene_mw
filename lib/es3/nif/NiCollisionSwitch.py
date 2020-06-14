from __future__ import annotations

from es3.utils.flags import bool_property
from .NiNode import NiNode


class NiCollisionSwitch(NiNode):

    # flags access
    propagate = bool_property(mask=0x0020)
