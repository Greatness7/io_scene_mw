from __future__ import annotations

from es3.utils.flags import bool_property
from .NiParticleSystemController import NiParticleSystemController


class NiBSPArrayController(NiParticleSystemController):

    # flags access
    at_vertices = bool_property(mask=0x0010)
