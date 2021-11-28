from __future__ import annotations

from .NiMeta import NiMeta


class NiObject(metaclass=NiMeta):

    _refs = ()  # type: tuple[str, ...]
    _ptrs = ()  # type: tuple[str, ...]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"{self.type}<{id(self)}>"

    def __getattr__(self, name):
        if not hasattr(self.defaults, name):
            raise AttributeError(f"'{self.type}' has no attribute '{name}'")
        value = getattr(self.defaults, name)()
        setattr(self, name, value)
        return value

    def load(self, stream):
        pass

    def save(self, stream):
        pass

    def sort(self, key=None):
        pass

    def apply_scale(self, scale):
        pass

    def _links(self):
        for name in self._refs:
            item = getattr(self, name)
            if item is None:
                continue
            if isinstance(item, NiObject):
                yield item
            else:
                yield from item

    def _traverse(self, seen) -> Iterator[NiObject]:
        seen.add(self)
        yield self
        for link in self._links():
            if link not in seen:
                yield from link._traverse(seen)

    def _resolve_links(self, objects):
        for name in self._refs + self._ptrs:
            item = getattr(self, name)
            if item is None:
                continue
            if type(item) is int:
                setattr(self, name, objects.get(item) if item != -1 else None)
            else:  # list of ints
                for index, value in enumerate(item):
                    item[index] = objects.get(value) if value != -1 else None

    @classmethod
    def attributes(cls):
        return {s for c in cls.__mro__[-2::-1] for s in c.__slots__}

    def _astuple(self, digits=4, ignore=()):
        # TODO atm this is only valid for NiProperty/NiSourceTexture/TexturingPropertyMap
        #   objects held in attributes are NOT converted to tuples... Should they be?
        fields = [("type", self.type)]
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


if __name__ == "__main__":
    from es3.utils.typing import *
