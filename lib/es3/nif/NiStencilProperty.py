from __future__ import annotations

from enum import IntEnum

from .NiProperty import NiProperty


class TestFunction(IntEnum):
    TEST_NEVER = 0
    TEST_LESS = 1
    TEST_EQUAL = 2
    TEST_LESSEQUAL = 3
    TEST_GREATER = 4
    TEST_NOTEQUAL = 5
    TEST_GREATEREQUAL = 6
    TEST_ALWAYS = 7


class Action(IntEnum):
    ACTION_KEEP = 0
    ACTION_ZERO = 1
    ACTION_REPLACE = 2
    ACTION_INCREMENT = 3
    ACTION_DECREMENT = 4
    ACTION_INVERT = 5


class DrawMode(IntEnum):
    DRAW_CCW_OR_BOTH = 0
    DRAW_CCW = 1
    DRAW_CW = 2
    DRAW_BOTH = 3


class NiStencilProperty(NiProperty):
    stencil_enabled: uint8 = 0
    stencil_function: int32 = TestFunction.TEST_NEVER
    stencil_ref: uint32 = 0
    stencil_mask: uint32 = 0xFFFFFFFF
    fail_action: int32 = Action.ACTION_KEEP
    pass_z_fail_action: int32 = Action.ACTION_KEEP
    pass_action: int32 = Action.ACTION_KEEP
    draw_mode: int32 = DrawMode.DRAW_CCW_OR_BOTH

    # provide access to related enums
    TestFunction = TestFunction
    Action = Action
    DrawMode = DrawMode

    def load(self, stream):
        super().load(stream)
        self.stencil_enabled = stream.read_ubyte()
        self.stencil_function = TestFunction(stream.read_int())
        self.stencil_ref = stream.read_uint()
        self.stencil_mask = stream.read_uint()
        self.fail_action = Action(stream.read_int())
        self.pass_z_fail_action = Action(stream.read_int())
        self.pass_action = Action(stream.read_int())
        self.draw_mode = DrawMode(stream.read_int())

    def save(self, stream):
        super().save(stream)
        stream.write_ubyte(self.stencil_enabled)
        stream.write_int(self.stencil_function)
        stream.write_uint(self.stencil_ref)
        stream.write_uint(self.stencil_mask)
        stream.write_int(self.fail_action)
        stream.write_int(self.pass_z_fail_action)
        stream.write_int(self.pass_action)
        stream.write_int(self.draw_mode)


if __name__ == "__main__":
    from es3.utils.typing import *
