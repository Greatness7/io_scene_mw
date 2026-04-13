from __future__ import annotations

from ..utils.math import np
from ..utils.typing import *


class NiMeta(type):
    __slots__ = ()
    defaults: type

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

    def __setattr__(cls, name: str, value: object) -> None:
        if hasattr(cls.defaults, name):
            # prevent assignments that would interfere with defaults
            raise TypeError(f"can't set attribute '{name}' of {cls}")
        super().__setattr__(name, value)


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


def _defaults_getter(value: Any) -> Callable[[], Any]:
    # `default` and `cls` are bound before the isinstance checks, as narrowing
    # `value` would only produce types that strict mode sees as partly unknown
    default: Any = value
    cls: type[Any] = default.__class__

    if isinstance(value, np.ndarray):
        return default.copy

    if isinstance(value, _hashable_types):
        return lambda: default

    return cls


def _create_defaults(name: str, bases: tuple[type, ...], defaults_dict: dict[str, Any]):
    _bases = tuple(b.defaults for b in bases if isinstance(b, NiMeta))
    _dict = {k: _defaults_getter(v) for k, v in defaults_dict.items()}
    return type(name + "Defaults", _bases, _dict)
