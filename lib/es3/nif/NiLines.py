from __future__ import annotations

from .NiGeometry import NiGeometry


class NiLines(NiGeometry):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError  # TODO
