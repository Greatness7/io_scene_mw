from __future__ import annotations

from es3.utils.math import np


class NiMeta(type):
    __slots__ = ()

    def __new__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]):

        # store the type name for serialization
        namespace.setdefault("type", name)

        # ensure the annotations mapping exists
        _annotations = namespace.setdefault("__annotations__", {})

        # derive slots via annotated attributes
        namespace["__slots__"] = _annotations

        # extract default values from namespace
        defaults = {k: namespace.pop(k) for k in _annotations}

        # create an accessor class for defaults
        namespace["defaults"] = _create_defaults(name, bases, defaults)

        return super().__new__(cls, name, bases, namespace)

    def __setattr__(cls, name, value):
        if hasattr(cls.defaults, name):
            # prevent assignments that would interfere with defaults
            raise TypeError(f"can't set attribute '{name}' of {cls}")
        super().__setattr__(name, value)


if __name__ == "__main__":
    from es3.utils.typing import *


_hashable_types = (
    int,
    float,
    type(None),
    str,
    bool,
    bytes,
    complex,
    frozenset,
    range,
    slice,
    tuple,
    type,
    type(Ellipsis),
    type(NotImplemented),
)
_defaults_cache: dict[tuple, Callable[[], Any]] = {}


def _defaults_getter(value: T) -> Callable[[], T]:

    if isinstance(value, np.ndarray):
        _hash = value.dtype, value.tobytes()
        return _defaults_cache.setdefault(_hash, value.copy)

    if isinstance(value, _hashable_types):
        _hash = type(value), value
        return _defaults_cache.setdefault(_hash, lambda: value)

    return type(value)


def _create_defaults(name: str, bases: tuple[type, ...], defaults_dict: dict[str, Any]):
    _bases = tuple(b.defaults for b in bases if isinstance(b, NiMeta))
    _dict = {k: _defaults_getter(v) for k, v in defaults_dict.items()}
    return type(name + "Defaults", _bases, _dict)
