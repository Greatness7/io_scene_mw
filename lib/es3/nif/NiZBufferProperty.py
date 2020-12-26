from __future__ import annotations

from enum import IntEnum

from es3.utils.flags import bool_property, enum_property
from .NiProperty import NiProperty


class TestFunction(IntEnum):
    TEST_ALWAYS = 0
    TEST_LESS = 1
    TEST_EQUAL = 2
    TEST_LESSEQUAL = 3
    TEST_GREATER = 4
    TEST_NOTEQUAL = 5
    TEST_GREATEREQUAL = 6
    TEST_NEVER = 7


class NiZBufferProperty(NiProperty):

    # provide access to related enums
    TestFunction = TestFunction

    # flags access
    z_buffer_test = bool_property(mask=0x0001)
    z_buffer_write = bool_property(mask=0x0002)
    test_function = enum_property(TestFunction, mask=0x003C, pos=2)
    test_function_specified = bool_property(mask=0x0040)
