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

        # calculate time deltas
        dt = times / 3.0
        if len(dt) >= 2:
            dt[:-1] = dt[1:] - dt[:-1]  # faster than np.diff(dt, append=0)
            dt[-1] = dt[-2]

        # control point handles
        shape = (2, *values.shape[::-1], 2)
        handles = np.empty(shape, dt.dtype)

        if handles.size:
            # incoming handles
            handles[0, ..., 0] = (times - dt)
            handles[0, ..., 1] = (values - self.in_tans / 3.0).T
            # outgoing handles
            handles[1, ..., 0] = (times + dt)
            handles[1, ..., 1] = (values - self.out_tans / 3.0).T

        return handles

    def get_tcb_tangent_handles(self):
        times, values = self.times, self.values

        # calculate deltas
        k = self.keys[:, :-3] / 3.0
        p = k - np.roll(k, +1, axis=0)
        n = np.roll(k, -1, axis=0) - k
        if len(k) >= 2:  # fix up ends
            p[0], n[-1] = p[1], n[-2]

        # calculate tangents
        mt, mc, mb = 1.0 - self.tcb.T
        pt, pc, pb = 1.0 + self.tcb.T

        in_tans = 0.5 * (
            (mt * mc * pb)[:, None] * p +
            (mt * pc * mb)[:, None] * n
        )
        out_tans = 0.5 * (
            (mt * pc * pb)[:, None] * p +
            (mt * mc * mb)[:, None] * n
        )

        # control point handles
        shape = (2, *values.shape[::-1], 2)
        handles = np.empty(shape, values.dtype)

        if handles.size:
            # incoming handles
            handles[0, ..., 0] = (times - in_tans[:, 0])
            handles[0, ..., 1] = (values - in_tans[:, 1:]).T
            # outgoing handles
            handles[1, ..., 0] = (times + out_tans[:, 0])
            handles[1, ..., 1] = (values + out_tans[:, 1:]).T

        return handles


if __name__ == "__main__":
    from es3.utils.typing import *
