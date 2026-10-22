from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils.typing import *

from .NiMeta import NiMeta

if TYPE_CHECKING:
    from .NiBinaryStream import NiBinaryStream


class NiObject(metaclass=NiMeta):
    _refs = ()
    _ptrs = ()
    _resolve_exclude = ()

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return f"{self.type}<{id(self)}>"

    def __getattr__(self, name: str) -> Any:
        if not hasattr(self.defaults, name):
            raise AttributeError(f"'{self.type}' has no attribute '{name}'")
        value = getattr(self.defaults, name)()
        setattr(self, name, value)
        return value

    def load(self, stream: NiBinaryStream) -> None:
        pass

    def save(self, stream: NiBinaryStream) -> None:
        pass

    def sort(self, key: Callable[[Any], Any] = lambda value: value) -> None:
        pass

    def apply_scale(self, scale: float) -> None:
        pass

    def apply_time_scale(self, scale: float):
        pass

    def _links(self) -> Iterator[NiObject | None]:
        for name in self._refs:
            item = getattr(self, name)
            if item is None:
                continue
            if isinstance(item, NiObject):
                yield item
            else:
                yield from item

    def traverse(self, seen: set[NiObject]) -> Iterator[NiObject]:
        seen.add(self)
        yield self
        for link in self._links():
            if link is not None and link not in seen:
                yield from link.traverse(seen)

    def resolve_links(self, objects: dict[int, NiObject]) -> None:
        for name in self._refs + self._ptrs:
            if name in self._resolve_exclude:
                continue
            item = getattr(self, name)
            if item is None:
                continue
            if type(item) is int:
                setattr(self, name, objects.get(item) if item != -1 else None)
            else:  # list of ints
                for index, value in enumerate(item):
                    item[index] = objects.get(value) if value != -1 else None

    @classmethod
    def attributes(cls) -> set[str]:
        return {name for base in cls.__mro__[-2::-1] for name in getattr(base, "__slots__", ())}

    def asdict(self) -> dict[str, object]:
        return {attr: getattr(self, attr) for attr in self.attributes()}

    def astuple(self, digits: int = 4, ignore: Collection[str] = ()) -> tuple[tuple[str, object], ...]:
        # TODO atm this is only valid for NiProperty/NiSourceTexture/TexturingPropertyMap
        #   objects held in attributes are NOT converted to tuples... Should they be?
        fields: list[tuple[str, object]] = [("type", self.type)]
        for name in self.attributes().difference(ignore):
            value = getattr(self, name)  # TODO attrgetter?
            if value is None:
                pass
            elif isinstance(value, (bool, int, str)):
                pass
            elif isinstance(value, NiObject):
                pass
            elif isinstance(value, float):
                value = round(value, digits)
            elif hasattr(value, "__array__"):
                value = tuple(value.round(digits).ravel().tolist())
            else:
                raise TypeError(f"astuple : unhandled type {type(value)}")

            fields.append((name, value))

        return tuple(fields)
