from __future__ import annotations

from .NiParticleModifier import NiParticleModifier


class NiParticleColorModifier(NiParticleModifier):
    color_data: NiColorData | None = None

    _refs = (*NiParticleModifier._refs, "color_data")

    def load(self, stream):
        super().load(stream)
        self.color_data = stream.read_link()

    def save(self, stream):
        super().save(stream)
        stream.write_link(self.color_data)


if __name__ == "__main__":
    from es3.nif import NiColorData
    from es3.utils.typing import *
