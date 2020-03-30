from __future__ import annotations

from .NiProperty import NiProperty

# TODO: enums


class NiStencilProperty(NiProperty):
    stencil_enabled: uint8 = 0
    stencil_function: uint32 = 0
    stencil_ref: uint32 = 0
    stencil_mask: uint32 = 0xFFFFFFFF
    fail_action: uint32 = 0
    z_fail_action: uint32 = 0
    pass_action: uint32 = 0
    draw_mode: uint32 = 3

    def load(self, stream):
        super().load(stream)
        self.stencil_enabled = stream.read_ubyte()
        self.stencil_function = stream.read_uint()
        self.stencil_ref = stream.read_uint()
        self.stencil_mask = stream.read_uint()
        self.fail_action = stream.read_uint()
        self.z_fail_action = stream.read_uint()
        self.pass_action = stream.read_uint()
        self.draw_mode = stream.read_uint()

    def save(self, stream):
        super().save(stream)
        stream.write_ubyte(self.stencil_enabled)
        stream.write_uint(self.stencil_function)
        stream.write_uint(self.stencil_ref)
        stream.write_uint(self.stencil_mask)
        stream.write_uint(self.fail_action)
        stream.write_uint(self.z_fail_action)
        stream.write_uint(self.pass_action)
        stream.write_uint(self.draw_mode)


if __name__ == "__main__":
    from es3.utils.typing import *
