from __future__ import annotations

from enum import IntEnum

from es3.utils.math import np, zeros
from .NiObject import NiObject


class KeyType(IntEnum):
    NO_INTERP = 0
    LIN_KEY = 1
    BEZ_KEY = 2
    TCB_KEY = 3
    EULER_KEY = 4


class NiFloatData(NiObject):
    key_type: int32 = KeyType.LIN_KEY
    keys: ndarray = zeros(0, 2)

    # provide access to related enums
    KeyType = KeyType

    def load(self, stream):
        num_keys = stream.read_uint()
        if num_keys:
            self.key_type = KeyType(stream.read_int())
            self.keys = stream.read_floats(num_keys, self.key_size)

    def save(self, stream):
        num_keys = len(self.keys)
        stream.write_uint(num_keys)
        if num_keys:
            stream.write_int(self.key_type)
            stream.write_floats(self.keys)

    @property
    def times(self) -> ndarray:
        return self.keys[:, 0]

    @property
    def values(self) -> ndarray:
        return self.keys[:, 1]

    @property
    def in_tans(self) -> ndarray:
        return self.keys[:, 2]

    @property
    def out_tans(self) -> ndarray:
        return self.keys[:, 3]

    @property
    def tcb(self) -> ndarray:
        return self.keys[:, -3:]

    @property
    def key_size(self) -> int:
        if self.key_type == KeyType.LIN_KEY:
            return 2  # (time, value)
        if self.key_type == KeyType.BEZ_KEY:
            return 4  # (time, value, inTan, outTan)
        if self.key_type == KeyType.TCB_KEY:
            return 5  # (time, value, tension, continuity, bias)
        raise Exception(f"{self.type} does not support '{self.key_type}'")

    def get_start_stop_times(self) -> tuple[int, int]:
        if len(self.keys) == 0:
            return (0, 0)
        else:
            return (self.keys[0, 0], self.keys[-1, 0])

    def get_tangent_handles(self):
        if self.key_type == KeyType.BEZ_KEY:
            return self.get_bez_tangent_handles()
        if self.key_type == KeyType.TCB_KEY:
            return self.get_tcb_tangent_handles()

    def get_bez_tangent_handles(self):
        times, values = self.times, self.values

        # control point handles
        shape = (2, *values.shape[::-1], 2)
        handles = np.empty(shape, values.dtype)

        if len(handles):
            dt = np.diff(times / 3.0, prepend=0, append=0)
            dt[0], dt[-1] = dt[1], dt[-2]  # correct edges

            # relative horizontal coordinates
            in_dx = dt[:-1]
            out_dx = dt[1:]

            # relative vertical coordinates
            in_dy = self.in_tans.T / 3.0
            out_dy = self.out_tans.T / 3.0

            # incoming handles
            handles[0, ..., 0] = times - in_dx
            handles[0, ..., 1] = values.T - in_dy
            # outgoing handles
            handles[1, ..., 0] = times + out_dx
            handles[1, ..., 1] = values.T - out_dy

        return handles

    def get_tcb_tangent_handles(self):
        times, values = self.times, self.values

        # control point handles
        shape = (2, *values.shape[::-1], 2)
        handles = np.empty(shape, values.dtype)

        if len(handles):
            # calculate deltas
            dx = (np.roll(times, 1, axis=0) - np.roll(times, -1, axis=0)) / 6.0
            dy = (np.roll(values, 1, axis=0) - np.roll(values, -1, axis=0)) / 6.0
            # fix up start/end
            dy[0] = dy[-1] = 0

            # TODO: tcb params
            # removed for now as they aren't supported by blender
            # instead this currently returns a Catmull–Rom Spline

            # incoming handles
            handles[0, ..., 0] = times - dx
            handles[0, ..., 1] = values.T - dy.T
            # outgoing handles
            handles[1, ..., 0] = times + dx
            handles[1, ..., 1] = values.T + dy.T

        return handles


if __name__ == "__main__":
    from es3.utils.typing import *
