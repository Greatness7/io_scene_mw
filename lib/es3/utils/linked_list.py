from __future__ import annotations


class LinkedListHelper:
    __slots__ = "name",

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, _):
        return LinkedList(instance, self.name)


class LinkedList:
    __slots__ = "owner", "name"

    def __init__(self, owner, name):
        self.owner = owner
        self.name = name

    def __iter__(self):
        node = self.head
        while node is not None:
            yield node
            node = node.next

    def __repr__(self):
        return repr(list(self))

    @property
    def head(self):
        return getattr(self.owner, self.name)

    @head.setter
    def head(self, item):
        setattr(self.owner, self.name, item)

    @property
    def tail(self):
        tail = None
        for tail in self:
            pass
        return tail

    def clear(self):
        for node in self:
            node.next = None
        self.head = None

    def append(self, item):
        if self.head is None:
            self.head = item
        else:
            self.tail.next = item

    def appendleft(self, item):
        item.next = self.head
        self.head = item

    def extend(self, items):
        tail = self.tail
        for item in items:
            if tail is None:
                self.head = item
            else:
                tail.next = item
            tail = item

    def extendleft(self, items):
        for item in items:
            self.appendleft(item)

    def pop(self):
        if self.head is None:
            raise ValueError("pop: called on empty linked list.")
        for owner, item in self.iter_owners():
            pass
        if owner is self.owner:
            self.head = None
        else:
            owner.next = None
        item.next = None
        return item

    def popleft(self):
        item = self.head
        if item is None:
            raise ValueError("popleft: called on empty linked list.")
        self.head = item.next
        item.next = None
        return item

    def remove(self, item):
        if item is self.head:
            owner = self.owner
            self.popleft()
        else:
            owner = self.find_owner(item)
            if owner is None:
                raise ValueError(f"remove: 'item' ({item}) was not found.")
            owner.next = item.next
            item.next = None
        return owner

    def insert_before(self, before, item):
        if before is self.head:
            self.appendleft(item)
            return
        owner = self.find_owner(before)
        if owner is None:
            raise ValueError(f"insert_before: 'before' ({before}) was not found.")
        owner.next = item
        item.next = before

    @staticmethod
    def insert_after(after, item):
        if after is None:
            raise ValueError("insert_after: 'after' value must not be None.")
        item.next = after.next
        after.next = item

    def iter_owners(self):
        owner, node = self.owner, self.head
        while node is not None:
            yield owner, node
            owner, node = node, node.next

    def find_owner(self, item):
        for owner, node in self.iter_owners():
            if node is item:
                return owner

    def find_type_with_owner(self, item_type):
        for owner, item in self.iter_owners():
            if isinstance(item, item_type):
                return owner, item

    def find_type(self, item_type: type[T]) -> T:
        result = self.find_type_with_owner(item_type)
        if result is not None:
            _, item = result
            return item

    def discard_type(self, item_type):
        result = self.find_type_with_owner(item_type)
        if result is not None:
            owner, item = result
            if owner is self.owner:
                self.popleft()
            else:
                owner.next = item.next
                item.next = None
            return item


if __name__ == "__main__":
    from es3.utils.typing import *
