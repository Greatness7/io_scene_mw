from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Generic, Protocol, TypeVar


class _LinkedNode(Protocol):
    next: Any


T = TypeVar("T", bound=_LinkedNode)
S = TypeVar("S", bound=_LinkedNode)


class LinkedListHelper(Generic[T]):
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, instance: object, _: type[object]) -> LinkedList[T]:
        return LinkedList(instance, self.name)


class LinkedList(Generic[T]):
    __slots__ = "owner", "name"

    def __init__(self, owner: object, name: str) -> None:
        self.owner = owner
        self.name = name

    def __iter__(self) -> Iterator[T]:
        node = self.head
        while node is not None:
            yield node
            node = node.next

    def __repr__(self) -> str:
        return repr(list(self))

    @property
    def head(self) -> T | None:
        return getattr(self.owner, self.name)

    @head.setter
    def head(self, item: T | None) -> None:
        setattr(self.owner, self.name, item)

    @property
    def tail(self) -> T | None:
        tail = None
        for tail in self:
            pass
        return tail

    def clear(self) -> None:
        for node in self:
            node.next = None
        self.head = None

    def append(self, item: T) -> None:
        tail = self.tail
        if tail is None:
            self.head = item
        else:
            tail.next = item

    def appendleft(self, item: T) -> None:
        item.next = self.head
        self.head = item

    def extend(self, items: Iterable[T]) -> None:
        tail = self.tail
        for item in items:
            if tail is None:
                self.head = item
            else:
                tail.next = item
            tail = item

    def extendleft(self, items: Iterable[T]) -> None:
        for item in items:
            self.appendleft(item)

    def pop(self) -> T:
        if self.head is None:
            raise ValueError("pop: called on empty linked list.")
        owner = self.owner
        item = self.head
        for owner, item in self.iter_owners():
            pass
        if owner is self.owner:
            self.head = None
        else:
            setattr(owner, "next", None)
        item.next = None
        return item

    def popleft(self) -> T:
        item = self.head
        if item is None:
            raise ValueError("popleft: called on empty linked list.")
        self.head = item.next
        item.next = None
        return item

    def remove(self, item: T) -> object | T:
        if item is self.head:
            owner = self.owner
            self.popleft()
        else:
            owner = self.find_owner(item)
            if owner is None:
                raise ValueError(f"remove: 'item' ({item}) was not found.")
            setattr(owner, "next", item.next)
            item.next = None
        return owner

    def insert_before(self, before: T, item: T) -> None:
        if before is self.head:
            self.appendleft(item)
            return
        owner = self.find_owner(before)
        if owner is None:
            raise ValueError(f"insert_before: 'before' ({before}) was not found.")
        setattr(owner, "next", item)
        item.next = before

    @staticmethod
    def insert_after(after: T | None, item: T) -> None:
        if after is None:
            raise ValueError("insert_after: 'after' value must not be None.")
        item.next = after.next
        after.next = item

    def iter_owners(self) -> Iterator[tuple[object | T, T]]:
        owner, node = self.owner, self.head
        while node is not None:
            yield owner, node
            owner, node = node, node.next

    def find_owner(self, item: T) -> object | T | None:
        for owner, node in self.iter_owners():
            if node is item:
                return owner

    def find_type_with_owner(self, item_type: type[S]) -> tuple[object | T, S] | None:
        for owner, item in self.iter_owners():
            if isinstance(item, item_type):
                return owner, item

    def find_type(self, item_type: type[S]) -> S | None:
        result = self.find_type_with_owner(item_type)
        if result is not None:
            _, item = result
            return item

    def discard_type(self, item_type: type[S]) -> S | None:
        result = self.find_type_with_owner(item_type)
        if result is not None:
            owner, item = result
            if owner is self.owner:
                self.popleft()
            else:
                setattr(owner, "next", item.next)
                item.next = None
            return item
