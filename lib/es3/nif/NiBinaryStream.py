from __future__ import annotations

from contextlib import contextmanager
from struct import pack, unpack

from es3.utils.io import BinaryStream
from es3.utils.math import np


class NiBinaryStream(BinaryStream):
    __slots__ = "history",

    def read_objects(self, object_types) -> Iterable[NiObject]:
        # prep history
        self.history = {}

        # read objects
        num_objects = self.read_uint()
        for i in range(num_objects):
            cls = object_types[self.read_str()]
            self.history[i] = self.read_type(cls)

        # resolve roots
        roots = [self.history.get(i) for i in self.read_links()]

        # resolve links
        for obj in self.history.values():
            obj._resolve_links(self.history)

            # fix roots
            if getattr(obj, "children", None):
                for i, root in enumerate(roots):
                    if root in obj.children:
                        roots[i] = obj

        # clear history
        del self.history

        return roots

    def write_objects(self, objects, roots):
        # prep history
        self.history = {}

        # build history
        for i, obj in enumerate(objects):
            self.history[obj] = i

        # write objects
        self.write_uint(len(self.history))
        for obj in self.history:
            self.write_str(obj.type)
            obj.save(self)

        # write roots
        self.write_links(roots)

        # clear history
        del self.history

    def read_bool(self) -> bool:
        return not not self.read_uint()

    def write_bool(self, value):
        self.write_uint(not not value)

    def read_link(self) -> int:
        return self.read_int()

    def write_link(self, obj: NiObject):
        self.write_int(self.history.get(obj, -1))

    def read_links(self) -> list[int]:
        length = self.read_uint()
        if length:
            links = self.read(length * 4)
            return list(unpack(f"<{length}i", links))
        return []

    def write_links(self, items: Collection[NiObject]):
        length = len(items)
        if length:
            links = [self.history.get(i, -1) for i in items]
            self.write(pack(f"<I{length}i", length, *links))
        else:
            self.write_uint(0)

    def read_type(self, cls: type[NiObject]) -> NiObject:
        obj = cls.__new__(cls)
        obj.load(self)
        return obj

    def write_type(self, obj: NiObject):
        obj.save(self)

    def read_array(self, shape, dtype):
        array = np.empty(shape, dtype)
        self.readinto(array)
        return array

    def write_array(self, array, dtype):
        if array.dtype != dtype:
            array = array.astype(dtype, copy=False)
        self.write(np.ascontiguousarray(array).view(np.ubyte))

    @staticmethod
    @contextmanager
    def reader(filepath: PathLike) -> Generator[NiBinaryStream, None, None]:
        with open(filepath, "rb") as f:
            data = f.read()
        with NiBinaryStream(data) as stream:
            yield stream

    @staticmethod
    @contextmanager
    def writer(filepath: PathLike) -> Generator[NiBinaryStream, None, None]:
        with NiBinaryStream() as stream:
            yield stream
            with open(filepath, "wb") as f:
                f.write(stream.getbuffer())


if __name__ == "__main__":
    from es3.nif import NiObject
    from es3.utils.typing import *
