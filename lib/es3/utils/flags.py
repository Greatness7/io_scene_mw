from __future__ import annotations


# noinspection PyPep8Naming
class bool_property:
    def __init__(self, mask):
        self.mask = mask

    def __get__(self, obj, cls):
        return (obj.flags & self.mask) == self.mask

    def __set__(self, obj, val):
        if val:
            obj.flags |= self.mask
        else:
            obj.flags &= ~self.mask


# noinspection PyPep8Naming
class enum_property:
    def __init__(self, enum, mask, pos):
        self.enum = enum
        self.mask = mask
        self.pos = pos

    def __get__(self, obj, cls):
        return self.enum((obj.flags & self.mask) >> self.pos).name

    def __set__(self, obj, val):
        obj.flags = (obj.flags & ~self.mask) | (self.enum[val] << self.pos)
