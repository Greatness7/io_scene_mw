from __future__ import annotations

from es3.utils.math import ZERO3, ZERO4
from .NiPerParticleData import NiPerParticleData
from .NiTimeController import NiTimeController


class NiParticleSystemController(NiTimeController):
    speed: float32 = 0.0
    speed_variation: float32 = 0.0
    declination_angle: float32 = 0.0  # [0, 2*pi]
    declination_variation: float32 = 0.0
    planar_angle: float32 = 0.0  # [0, 2*pi]
    planar_angle_variation: float32 = 0.0
    initial_normal: NiPoint3 = ZERO3
    initial_color: NiColorA = ZERO4
    initial_size: float32 = 0.0
    emit_start_time: float32 = 0.0
    emit_stop_time: float32 = 0.0
    reset_particle_system: uint8 = 0
    birth_rate: float32 = 0.0
    lifespan: float32 = 0.0
    lifespan_variation: float32 = 0.0
    use_birth_rate: uint8 = 1
    spawn_on_death: uint8 = 0
    emitter_width: float32 = 0.0
    emitter_height: float32 = 0.0
    emitter_depth: float32 = 0.0
    emitter: NiAVObject | None = None
    spawn_generations: uint16 = 0
    spawn_percentage: float32 = 0.0  # [0.0, 1.0]
    spawn_multiplier: uint16 = 1
    spawned_speed_chaos: float32 = 0.0
    spawned_direction_chaos: float32 = 0.0
    particles: list[NiPerParticleData] = []
    num_active_particles: uint16 = 0
    emitter_modifier: NiEmitterModifier | None = None
    particle_modifier: NiParticleModifier | None = None
    particle_collider: NiParticleCollider | None = None
    compute_dynamic_bounding_volume: uint8 = 0

    _refs = (*NiTimeController._refs, "emitter_modifier", "particle_modifier", "particle_collider")
    _ptrs = (*NiTimeController._ptrs, "emitter")

    def load(self, stream):
        super().load(stream)
        self.speed = stream.read_float()
        self.speed_variation = stream.read_float()
        self.declination_angle = stream.read_float()  # [0, 2*pi)
        self.declination_variation = stream.read_float()
        self.planar_angle = stream.read_float()  # [0, 2*pi)
        self.planar_angle_variation = stream.read_float()
        self.initial_normal = stream.read_floats(3)
        self.initial_color = stream.read_floats(4)
        self.initial_size = stream.read_float()
        self.emit_start_time = stream.read_float()
        self.emit_stop_time = stream.read_float()
        self.reset_particle_system = stream.read_ubyte()
        self.birth_rate = stream.read_float()
        self.lifespan = stream.read_float()
        self.lifespan_variation = stream.read_float()
        self.use_birth_rate = stream.read_ubyte()
        self.spawn_on_death = stream.read_ubyte()
        self.emitter_width = stream.read_float()
        self.emitter_height = stream.read_float()
        self.emitter_depth = stream.read_float()
        self.emitter = stream.read_link()
        self.spawn_generations = stream.read_ushort()
        self.spawn_percentage = stream.read_float()
        self.spawn_multiplier = stream.read_ushort()
        self.spawned_speed_chaos = stream.read_float()
        self.spawned_direction_chaos = stream.read_float()
        num_particles = stream.read_ushort()
        self.num_active_particles = stream.read_ushort()
        if num_particles:
            self.particles = [stream.read_type(NiPerParticleData) for _ in range(num_particles)]
        self.emitter_modifier = stream.read_link()
        self.particle_modifier = stream.read_link()
        self.particle_collider = stream.read_link()
        self.compute_dynamic_bounding_volume = stream.read_ubyte()

    def save(self, stream):
        super().save(stream)
        stream.write_float(self.speed)
        stream.write_float(self.speed_variation)
        stream.write_float(self.declination_angle)
        stream.write_float(self.declination_variation)
        stream.write_float(self.planar_angle)
        stream.write_float(self.planar_angle_variation)
        stream.write_floats(self.initial_normal)
        stream.write_floats(self.initial_color)
        stream.write_float(self.initial_size)
        stream.write_float(self.emit_start_time)
        stream.write_float(self.emit_stop_time)
        stream.write_ubyte(self.reset_particle_system)
        stream.write_float(self.birth_rate)
        stream.write_float(self.lifespan)
        stream.write_float(self.lifespan_variation)
        stream.write_ubyte(self.use_birth_rate)
        stream.write_ubyte(self.spawn_on_death)
        stream.write_float(self.emitter_width)
        stream.write_float(self.emitter_height)
        stream.write_float(self.emitter_depth)
        stream.write_link(self.emitter)
        stream.write_ushort(self.spawn_generations)
        stream.write_float(self.spawn_percentage)
        stream.write_ushort(self.spawn_multiplier)
        stream.write_float(self.spawned_speed_chaos)
        stream.write_float(self.spawned_direction_chaos)
        stream.write_ushort(len(self.particles))
        stream.write_ushort(self.num_active_particles)
        for item in self.particles:
            item.save(stream)
        stream.write_link(self.emitter_modifier)
        stream.write_link(self.particle_modifier)
        stream.write_link(self.particle_collider)
        stream.write_ubyte(self.compute_dynamic_bounding_volume)


if __name__ == "__main__":
    from es3.nif import NiAVObject, NiEmitterModifier, NiParticleCollider, NiParticleModifier
    from es3.utils.typing import *
