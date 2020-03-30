from __future__ import annotations

from .NiLight import NiLight


class NiPointLight(NiLight):
    constant_attenuation: float32 = 0.0
    linear_attenuation: float32 = 0.0
    quadratic_attenuation: float32 = 0.0

    def load(self, stream):
        super().load(stream)
        self.constant_attenuation = stream.read_float()
        self.linear_attenuation = stream.read_float()
        self.quadratic_attenuation = stream.read_float()

    def save(self, stream):
        super().save(stream)
        stream.write_float(self.constant_attenuation)
        stream.write_float(self.linear_attenuation)
        stream.write_float(self.quadratic_attenuation)


if __name__ == "__main__":
    from es3.utils.typing import *
