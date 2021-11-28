from __future__ import annotations

from enum import IntEnum

from es3.utils.math import np, quaternion_from_euler_angle, quaternion_mul, zeros
from .NiFloatData import KeyType, NiFloatData


class AxisOrder(IntEnum):
    XYZ = 0
    XZY = 1
    YZX = 2
    YXZ = 3
    ZXY = 4
    ZYX = 5
    XYX = 6
    YZY = 7
    ZXZ = 8


class NiRotData(NiFloatData):
    euler_axis_order: int32 = AxisOrder.XYZ
    euler_data: tuple[NiFloatData, ...] = ()

    # provide access to related enums
    AxisOrder = AxisOrder

    def load(self, stream):
        num_keys = stream.read_uint()
        if num_keys:
            self.key_type = KeyType(stream.read_int())
            if self.key_type == KeyType.EULER_KEY:
                self.euler_axis_order = AxisOrder(stream.read_int())
                self.euler_data = (stream.read_type(NiFloatData),
                                   stream.read_type(NiFloatData),
                                   stream.read_type(NiFloatData))
            else:
                self.keys = stream.read_floats(num_keys, self.key_size)

    def save(self, stream):
        num_keys = self._num_keys()
        stream.write_uint(num_keys)
        if num_keys:
            stream.write_int(self.key_type)
            if self.key_type == KeyType.EULER_KEY:
                stream.write_int(self.euler_axis_order)
                self.euler_data[0].save(stream)
                self.euler_data[1].save(stream)
                self.euler_data[2].save(stream)
            else:
                stream.write_floats(self.keys)

    @property
    def values(self) -> ndarray:
        return self.keys[:, 1:5]

    @property
    def in_tans(self) -> ndarray:
        raise IndexError

    @property
    def out_tans(self) -> ndarray:
        raise IndexError

    @property
    def tcb(self) -> ndarray:
        return self.keys[:, -3:]

    @property
    def key_size(self) -> int:
        if self.key_type == KeyType.LIN_KEY:
            return 5  # (time, w, x, y, z)
        if self.key_type == KeyType.BEZ_KEY:
            return 5  # (time, w, x, y, z)
        if self.key_type == KeyType.TCB_KEY:
            return 8  # (time, w, x, y, z, tension, continuity, bias)
        raise Exception(f"{self.type} does not support '{self.key_type}'")

    def _num_keys(self):
        if self.key_type == KeyType.EULER_KEY:
            return any(len(e.keys) for e in self.euler_data)
        return len(self.keys)

    def convert_to_quaternions(self):
        if self.euler_data == ():
            return  # already using quaternions

        # TODO: support alternative axis orders
        assert self.euler_axis_order == AxisOrder.XYZ

        # extract keys and clear euler settings
        e_keys = [e.keys for e in self.euler_data]
        del self.key_type, self.euler_data, self.euler_axis_order

        x, y, z = map(len, e_keys)
        if x == y == z == 0:
            return  # no keys exist on any axis

        q_keys = zeros(x+y+z, 5)
        slices = np.s_[:x, x:x+y, x+y:x+y+z]

        for i, keys in enumerate(e_keys):
            if len(keys) == 0:
                continue
            # get slice
            q = q_keys[slices[i]]
            # set times
            q[:, 0] = keys[:, 0]
            # set quats
            quaternion_from_euler_angle(angle=keys[:, 1], euler_axis=i, out=q[:, 1:5])

        # sort and combine keys of same timings
        u, i, v = np.unique(q_keys[:, 0], return_index=True, return_inverse=True)
        if len(u) == len(q_keys):
            self.keys = q_keys[i]
        else:
            self.keys = zeros(len(u), 5)
            self.keys[:, 0] = u
            self.keys[:, 1] = 1
            for index in i:
                q = self.keys[index, 1:]
                for other in q_keys[index == v, 1:]:
                    quaternion_mul(other, q, out=q)


if __name__ == "__main__":
    from es3.utils.typing import *
