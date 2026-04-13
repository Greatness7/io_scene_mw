from __future__ import annotations

from ..utils.flags import bool_property

from .NiBSAnimationNode import NiBSAnimationNode


class NiBSParticleNode(NiBSAnimationNode):

    # flags access
    follow = bool_property(mask=0x0080)
