from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

NodeId = TypeVar("NodeId", bound=Hashable)
Array = NDArray[Any]


def evaluate_world_matrices(records: Iterable[tuple[NodeId, NodeId | None, Array]]) -> dict[NodeId, Array]:
    """Evaluate independent world matrices from parent-before-child local transforms."""
    worlds: dict[NodeId, Array] = {}

    for node, parent, local in records:
        local_array = np.asarray(local)
        if parent is None:
            world = np.array(local_array, copy=True)
        else:
            try:
                parent_world = worlds[parent]
            except KeyError:
                raise ValueError(f"evaluate_world_matrices: parent {parent!r} of {node!r} has not been evaluated") from None
            world = np.array(parent_world @ local_array, copy=True)

        worlds[node] = world

    return worlds
