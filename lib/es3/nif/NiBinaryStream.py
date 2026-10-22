from __future__ import annotations

from contextlib import contextmanager
from struct import pack, unpack
from typing import TypeVar

from ..utils.io import BinaryStream
from ..utils.typing import *

from .NiObject import NiObject


NiObjectT = TypeVar("NiObjectT", bound=NiObject)


class NiBinaryStream(BinaryStream):
    __slots__ = "_read_history", "_write_history"

    _read_history: dict[int, NiObject]
    _write_history: dict[Any, int]  # we key as `Any` so that `None` look ups do not type error

    def read_objects(self, object_types: Mapping[str, type[NiObject]]) -> list[NiObject | None]:
        # prep history
        self._read_history = {}

        # read objects
        num_objects = self.read_uint()
        for i in range(num_objects):
            cls = object_types[self.read_str()]
            self._read_history[i] = self.read_type(cls)

        # resolve roots
        roots = [self._read_history.get(i) for i in self.read_links()]

        # resolve links
        for obj in self._read_history.values():
            obj.resolve_links(self._read_history)

            # fix roots: a declared root that is really a child of another object is replaced by that
            # parent. Compare non-null slots by identity -- `root in obj.children` matches an unresolved
            # root (`None`, from a dangling or -1 root link) against any null child slot.
            children = getattr(obj, "children", None)
            if children:
                for i, root in enumerate(roots):
                    if root is not None and any(child is root for child in children):
                        roots[i] = obj

        # clear history
        del self._read_history

        return roots

    def write_objects(self, objects: Iterable[NiObject], roots: Collection[NiObject | None]) -> None:
        # prep history
        self._write_history = {}

        # build history; the index must equal the row the object is written to, so take it from the
        # dict's own length rather than from the input. `objects` is public API and a caller may pass a
        # sequence containing repeats, which would otherwise record an index past the end of the table.
        for obj in objects:
            self._write_history.setdefault(obj, len(self._write_history))

        # write objects
        self.write_uint(len(self._write_history))
        for obj in self._write_history:
            self.write_str(obj.type)
            obj.save(self)

        # write roots
        self.write_links(roots)

        # clear history
        del self._write_history

    def read_bool(self) -> bool:
        return not not self.read_uint()

    def write_bool(self, value: object) -> None:
        self.write_uint(not not value)

    def read_link(self) -> Link:
        return self.read_int()

    def write_link(self, obj: NiObject | None) -> None:
        self.write_int(self._write_history.get(obj, -1))

    def read_links(self) -> list[Link]:
        length = self.read_uint()
        if length:
            links = self.read(length * 4)
            return list(unpack(f"<{length}i", links))
        return []

    def write_links(self, items: Collection[NiObject | None]) -> None:
        length = len(items)
        if length:
            links = [self._write_history.get(i, -1) for i in items]
            self.write(pack(f"<I{length}i", length, *links))
        else:
            self.write_uint(0)

    def read_type(self, cls: type[NiObjectT]) -> NiObjectT:
        obj = cls.__new__(cls)
        obj.load(self)
        return obj

    def write_type(self, obj: NiObject):
        obj.save(self)

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
