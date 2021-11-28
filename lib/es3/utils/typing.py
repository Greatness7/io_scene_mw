from typing import (
    AbstractSet,
    Any,
    AnyStr,
    AsyncContextManager,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    ByteString,
    Callable,
    ClassVar,
    Collection,
    Container,
    ContextManager,
    Coroutine,
    Counter,
    DefaultDict,
    Deque,
    # Dict,
    FrozenSet,
    Generator,
    Generic,
    Hashable,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    # List,
    Mapping,
    MappingView,
    MutableMapping,
    MutableSequence,
    MutableSet,
    NamedTuple,
    NewType,
    NoReturn,
    # Optional,
    Reversible,
    Sequence,
    # Set,
    Sized,
    SupportsAbs,
    SupportsBytes,
    SupportsComplex,
    SupportsFloat,
    SupportsInt,
    SupportsRound,
    Text,
    # Tuple,
    # Type,
    TYPE_CHECKING,
    TypeVar,
    # Union,
    ValuesView,
)

if TYPE_CHECKING:
    from numpy import ndarray
    from pathlib import Path

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

    T = TypeVar("T")
    PathLike = AnyStr | Path
