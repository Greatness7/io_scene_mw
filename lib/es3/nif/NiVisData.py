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

    @property
    def times(self) -> ndarray:
        return self.keys["f0"]

    @times.setter
    def times(self, array: ndarray):
        self.keys["f0"] = array

    @property
    def values(self) -> ndarray:
        return self.keys["f1"]

    @values.setter
    def values(self, array: ndarray):
        self.keys["f1"] = array

    def get_start_stop_times(self) -> tuple[int, int]:
        if len(self.keys) == 0:
            return (0, 0)
        else:
            return (self.times[0], self.times[-1])


if __name__ == "__main__":
    from es3.utils.typing import *
