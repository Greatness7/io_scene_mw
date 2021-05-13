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
    interpolation: int32 = KeyType.LIN_KEY
    keys: ndarray = zeros(0, 2)

    # provide access to related enums
    KeyType = KeyType

    def load(self, stream):
        num_keys = stream.read_uint()
        if num_keys:
            self.interpolation = KeyType(stream.read_int())
            self.keys = stream.read_floats(num_keys, self.key_size)

    def save(self, stream):
        num_keys = len(self.keys)
        stream.write_uint(num_keys)
        if num_keys:
            stream.write_int(self.interpolation)
            stream.write_floats(self.keys)

    @property
    def times(self):
        return self.keys[:, 0]

    @property
    def values(self):
        return self.keys[:, 1]

    @property
    def in_tans(self):
        return self.keys[:, 2]

    @property
    def out_tans(self):
        return self.keys[:, 3]

    @property
    def tcb(self):
        return self.keys[:, 1:4]

    @property
    def key_size(self):
        if self.interpolation == KeyType.LIN_KEY:
            return 2  # (time, value)
        if self.interpolation == KeyType.BEZ_KEY:
            return 4  # (time, value, inTan, outTan)
        if self.interpolation == KeyType.TCB_KEY:
            return 5  # (time, value, tension, continuity, bias)
        raise Exception(f"{self.type} does not support '{self.interpolation}'")

    def get_start_stop_times(self):
        if len(self.keys) == 0:
            return (0, 0)
        else:
            return (self.keys[0, 0], self.keys[-1, 0])

    def get_tangent_handles(self):
        if self.interpolation == KeyType.BEZ_KEY:
            return self.get_bez_tangent_handles()
        if self.interpolation == KeyType.TCB_KEY:
            return self.get_tcb_tangent_handles()

    def get_bez_tangent_handles(self):
        times, values = self.times, self.values

        # relative horizontal coordinates
        dt = np.pad(np.diff(times), (1, 1), mode='edge') / 3.0
        in_dx = dt[:-1]
        out_dx = dt[1:]

        # relative vertical coordinates
        in_dy = self.in_tans.T / 3.0
        out_dy = self.out_tans.T / 3.0

        # control point handles
        shape = (2, *values.shape[::-1], 2)
        handles = np.empty(shape, dt.dtype)

        if handles.size:
            # incoming handles
            handles[0, ..., 0] = times - in_dx
            handles[0, ..., 1] = values.T - in_dy
            # outgoing handles
            handles[1, ..., 0] = times + out_dx
            handles[1, ..., 1] = values.T + out_dy

        return handles

    def get_tcb_tangent_handles(self):
        times, values = self.times, self.values

        # calculate deltas
        dt = np.pad(np.diff(times), (1, 1), mode='edge') / 3.0
        in_dx = dt[:-1]
        out_dx = dt[1:]

        # calculate tangents
        mt, mc, mb = 1.0 - self.tcb.T
        pt, pc, pb = 1.0 + self.tcb.T

        in_tans = 0.5 * (
            (mt * mc * pb)[:, None] * in_dx +
            (mt * pc * mb)[:, None] * out_dx
        )
        out_tans = 0.5 * (
            (mt * pc * pb)[:, None] * in_dx +
            (mt * mc * mb)[:, None] * out_dx
        )

        # control point handles
        shape = (2, *values.shape[::-1], 2)
        handles = np.empty(shape, values.dtype)

        if handles.size:
            # incoming handles
            handles[0, ..., 0] = times - in_tans[:, 0]
            handles[0, ..., 1] = values.T - in_tans[:, 1:]
            # outgoing handles
            handles[1, ..., 0] = times + out_tans[:, 0]
            handles[1, ..., 1] = values.T + out_tans[:, 1:]

        return handles


if __name__ == "__main__":
    from es3.utils.typing import *
