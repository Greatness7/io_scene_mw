from __future__ import annotations

from es3.utils.math import np, zeros
from .NiObject import NiObject

_dtype = np.dtype("<f, <B")


class NiVisData(NiObject):
    keys: ndarray = zeros(0, dtype=_dtype)

    def load(self, stream):
        num_keys = stream.read_uint()
        if num_keys:
            self.keys = stream.read_array(num_keys, _dtype)

    def save(self, stream):
        stream.write_uint(len(self.keys))
        stream.write_array(self.keys, _dtype)


if __name__ == "__main__":
    from es3.utils.typing import *
