from __future__ import annotations

from .NiPointLight import NiPointLight


class NiSpotLight(NiPointLight):
    outer_spot_angle: float32 = 0.0
    exponent: float32 = 0.0

    def load(self, stream):
        super().load(stream)
        self.outer_spot_angle = stream.read_float()
        self.exponent = stream.read_float()

    def save(self, stream):
        super().save(stream)
        stream.write_float(self.outer_spot_angle)
        stream.write_float(self.exponent)


if __name__ == "__main__":
    from es3.utils.typing import *
