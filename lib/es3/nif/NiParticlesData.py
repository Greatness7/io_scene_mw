from __future__ import annotations

from es3.utils.math import zeros
from .NiGeometryData import NiGeometryData


class NiParticlesData(NiGeometryData):
    num_particles: uint16 = 0
    particle_radius: float32 = 0.0
    num_active: uint16 = 0
    sizes: ndarray = zeros(0)

    def load(self, stream):
        super().load(stream)
        self.num_particles = stream.read_ushort()
        self.particle_radius = stream.read_float()
        self.num_active = stream.read_ushort()
        has_sizes = stream.read_bool()
        if has_sizes:
            self.sizes = stream.read_floats(len(self.vertices))

    def save(self, stream):
        super().save(stream)
        stream.write_ushort(self.num_particles)
        stream.write_float(self.particle_radius)
        stream.write_ushort(self.num_active)
        num_sizes = len(self.sizes)
        stream.write_bool(num_sizes)
        if num_sizes:
            stream.write_floats(self.sizes)


if __name__ == "__main__":
    from es3.utils.typing import *
