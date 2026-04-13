from __future__ import annotations

from enum import IntEnum
from typing import Generic, Protocol, TypeVar


class _FlagOwner(Protocol):
    flags: int


EnumT = TypeVar("EnumT", bound=IntEnum)


class bool_property:
    __slots__ = ("mask",)

    def __init__(self, mask: int) -> None:
        self.mask = mask

    def __get__(self, obj: _FlagOwner, cls: type[_FlagOwner]) -> bool:
        return (obj.flags & self.mask) == self.mask

    def __set__(self, obj: _FlagOwner, val: bool) -> None:
        if val:
            obj.flags |= self.mask
        else:
            obj.flags &= ~self.mask


class enum_property(Generic[EnumT]):
    __slots__ = "enum", "mask", "pos"

    def __init__(self, enum: type[EnumT], mask: int, pos: int) -> None:
        self.enum = enum
        self.mask = mask
        self.pos = pos

    def __get__(self, obj: _FlagOwner, cls: type[_FlagOwner]) -> str:
        return self.enum((obj.flags & self.mask) >> self.pos).name

    def __set__(self, obj: _FlagOwner, val: str) -> None:
        obj.flags = (obj.flags & ~self.mask) | (self.enum[val] << self.pos)
