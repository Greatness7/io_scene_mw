from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Protocol, overload


class _ChildOwner(Protocol):
    _children: list[Any]


class ChildrenView(Sequence[Any]):
    """Ordered, read-only window onto a node's owned child slots.

    Reads the owner's current backing list on every access, so a view obtained before `replace_children`
    can never hand back the list that was replaced. Slots may be `None`; membership is by identity.
    Deliberately defines no `__eq__`: comparing against a list must not silently succeed, because callers
    that want a snapshot should say `list(node.children)`.
    """

    __slots__ = ("_owner",)

    def __init__(self, owner: _ChildOwner) -> None:
        self._owner = owner

    @property
    def _items(self) -> list[Any]:
        return self._owner._children  # pyright: ignore[reportPrivateUsage]

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> Any: ...

    @overload
    def __getitem__(self, index: slice) -> list[Any]: ...

    def __getitem__(self, index: int | slice) -> Any:
        return self._items[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __reversed__(self) -> Iterator[Any]:
        return reversed(self._items)

    def __contains__(self, value: object) -> bool:
        return any(item is value for item in self._items)

    def __repr__(self) -> str:
        return repr(self._items)
