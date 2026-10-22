# `X as X` marks a re-export, otherwise strict pyright flags unused imports.
from typing import (
    Any,
    Callable as Callable,
    Collection as Collection,
    Generator as Generator,
    Iterable as Iterable,
    Iterator as Iterator,
    Mapping as Mapping,
    Sequence as Sequence,
    TYPE_CHECKING,
    TypeVar,
)

if TYPE_CHECKING:
    import os

    from numpy import ndarray

    # aliases
    int8 = int
    uint8 = int
    int16 = int
    uint16 = int
    int32 = int
    uint32 = int
    float32 = float

    # nif aliases
    NiPoint2 = ndarray  # float[2]
    NiPoint3 = ndarray  # float[3]
    NiColor = ndarray  # float[3]
    NiColorA = ndarray  # float[4]
    NiPlane = ndarray  # float[4]
    NiRect = ndarray  # float[4]
    NiFrustum = ndarray  # float[6]
    NiMatrix2 = ndarray  # float[2, 2]
    NiMatrix3 = ndarray  # float[3, 3]

    # An unresolved object reference. On disk a link is an index into the object
    # table. The `resolve_links` function swaps it for the object (or None) once
    # every object has been read.
    Link = Any

    T = TypeVar("T")
    PathLike = str | os.PathLike[str]
