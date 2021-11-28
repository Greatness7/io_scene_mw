from __future__ import annotations

from .NiObject import NiObject


class NiParticleModifier(NiObject):
    next: NiParticleModifier | None = None
    controller: NiParticleSystemController | None = None

    _refs = (*NiObject._refs, "next")
    _ptrs = (*NiObject._refs, "controller")

    def load(self, stream):
        self.next = stream.read_link()
        self.controller = stream.read_link()

    def save(self, stream):
        stream.write_link(self.next)
        stream.write_link(self.controller)


if __name__ == "__main__":
    from es3.nif import NiParticleSystemController
    from es3.utils.typing import *
