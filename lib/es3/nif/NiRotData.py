# NumPy's overloads read as partially unknown under strict mode; the arrays here are typed explicitly.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

from ..utils.math import (
    np,
    quaternion_conjugate,
    quaternion_exp,
    quaternion_from_euler_angle,
    quaternion_log,
    quaternion_mul,
    quaternion_slerp,
    quaternion_squad,
    quaternion_squad_control,
    zeros,
)
from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiFloatData import KeyType, NiFloatData

if TYPE_CHECKING:
    from ..utils.math import Array


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


# The axes each euler order applies, first to last (data_NiEulerRotKey_permutations 0x7c5b88).
EULER_AXIS_SEQUENCES = ((0, 1, 2), (0, 2, 1), (1, 2, 0), (1, 0, 2), (2, 0, 1), (2, 1, 0), (0, 1, 0), (1, 2, 1), (2, 0, 2))


class NiRotData(NiFloatData):
    euler_axis_order: int32 = AxisOrder.XYZ
    euler_data: tuple[NiFloatData, ...] = ()

    # provide access to related enums
    AxisOrder = AxisOrder

    def load(self, stream: NiBinaryStream) -> None:
        num_keys = stream.read_uint()
        if num_keys:
            self.key_type = KeyType(stream.read_int())
            if self.key_type == KeyType.EULER_KEY:
                self.euler_axis_order = AxisOrder(stream.read_int())
                self.euler_data = (
                    stream.read_type(NiFloatData),
                    stream.read_type(NiFloatData),
                    stream.read_type(NiFloatData),
                )
            else:
                self.keys = stream.read_floats(num_keys, self.key_size)

    def save(self, stream: NiBinaryStream) -> None:
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

        q_keys = zeros(x + y + z, 5)
        slices = np.s_[:x, x : x + y, x + y : x + y + z]

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
        u, i, v = np.unique(q_keys[:, 0], return_index=True, return_inverse=True)  # pyright: ignore[reportUnknownMemberType]
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

    def evaluate(self, times: Array) -> Array:
        """Rotations at `times` as (len(times), 4) quaternions (w, x, y, z), as the engine plays them.

        LIN keys slerp, BEZ and TCB keys squad, and euler keys combine their three axes in the stored order.
        Times outside the key range hold the end keys.
        """
        query = np.atleast_1d(np.asarray(times, dtype=self.keys.dtype))
        if self.key_type == KeyType.EULER_KEY:
            return self._evaluate_euler(query)

        n = len(self.keys)
        if n == 0:
            raise ValueError(f"{self.type} has no keys to evaluate")
        values = self._loaded_values()
        if n == 1:
            return np.repeat(values, len(query), axis=0)

        key_times = self.times
        segment, fraction = self._locate(key_times, query)
        p, q = values[segment], values[segment + 1]
        if self.key_type == KeyType.LIN_KEY:
            result = quaternion_slerp(p, q, fraction)
        else:
            a, _ = self._squad_controls(values, segment)
            _, b = self._squad_controls(values, segment + 1)
            result = quaternion_squad(p, a, b, q, fraction)

        # genInterp (0x70c5b0) returns the end keys themselves outside the key range
        result[query <= key_times[0]] = values[0]
        result[query >= key_times[-1]] = values[-1]
        return result

    def evaluate_and_insert(self, times: Array, atol: float = 1e-6) -> None:
        """Insert keys at `times` without changing the rotation. LIN keys split exactly along their slerp, and
        euler keys insert on each keyed axis. BEZ and TCB quaternion keys raise ValueError: their squad
        controls are derived from the neighbouring keys, so no inserted key leaves the curve unchanged.
        """
        if self.key_type == KeyType.EULER_KEY:
            for euler_data in self.euler_data:
                if len(euler_data.keys):
                    euler_data.evaluate_and_insert(times, atol)
            return
        if self.key_type != KeyType.LIN_KEY:
            raise ValueError(f"{self.type} {KeyType(self.key_type).name} keys cannot be split exactly")
        super().evaluate_and_insert(times, atol)

    def _loaded_values(self) -> Array:
        """The key quaternions as the engine holds them after loading (fillDerived_NoInterpolation 0x70c7b0).

        A key is negated when it faces away from the (already adjusted) key before it, then w is clamped
        into [-1, 1]. Slerp itself never takes the short way round, so this is what keeps motion short.
        """
        values: Array = np.array(self.values)
        if len(values) > 1:
            dots = np.einsum("ij,ij->i", values[:-1], values[1:])
            values[1:] *= np.cumprod(np.where(dots < 0, -1.0, 1.0))[:, None]
        np.clip(values[:, 0], -1.0, 1.0, out=values[:, 0])
        return values

    def _squad_controls(self, values: Array, index: Array) -> tuple[Array, Array]:
        """The squad controls of keys `index`: (a) leaving the key and (b) arriving at it.

        End keys stand in for their own missing neighbour (fillDerived_Bezier 0x714350, fillDerived_TCB
        0x720910).
        """
        last = len(values) - 1
        before, after = np.maximum(index - 1, 0), np.minimum(index + 1, last)
        prev, current, next = values[before], values[index], values[after]
        if self.key_type == KeyType.BEZ_KEY:
            control = quaternion_squad_control(prev, current, next)
            return control, control

        # TCB (0x7206d0): Kochanek-Bartels in log space, weighted by the key's position between its neighbours
        key_times: Array = self.times
        span = key_times[after] - key_times[before]
        lead = np.divide(key_times[index] - key_times[before], span, out=np.zeros_like(span), where=span > 0)[:, None]
        trail = np.divide(key_times[after] - key_times[index], span, out=np.zeros_like(span), where=span > 0)[:, None]

        tension, continuity, bias = (self.tcb[index, i, None] for i in range(3))
        loose = 1.0 - tension
        incoming = quaternion_log(quaternion_mul(quaternion_conjugate(prev), current))
        outgoing = quaternion_log(quaternion_mul(quaternion_conjugate(current), next))

        leaving = lead * loose * ((1 + bias) * (1 + continuity) * incoming + (1 - bias) * (1 - continuity) * outgoing)
        arriving = trail * loose * ((1 + bias) * (1 - continuity) * incoming + (1 - bias) * (1 + continuity) * outgoing)
        a = quaternion_mul(current, quaternion_exp(0.5 * (leaving - outgoing)))
        b = quaternion_mul(current, quaternion_exp(0.5 * (incoming - arriving)))
        return a, b

    def _evaluate_euler(self, query: Array) -> Array:
        """interpolate_Euler (0x715150): each axis evaluates on its own keys (angle 0 when it has none), then the
        axis rotations apply in the stored order."""
        angles = np.zeros((len(query), 3), self.keys.dtype)
        for i, euler_data in enumerate(self.euler_data):
            if len(euler_data.keys):
                angles[:, i] = euler_data.evaluate(query)[:, 0]

        axis_rotations = [zeros(len(query), 4) for _ in range(3)]
        for i, rotation in enumerate(axis_rotations):
            quaternion_from_euler_angle(angles[:, i], i, out=rotation)
        result = zeros(len(query), 4)
        result[:, 0] = 1.0
        for axis in EULER_AXIS_SEQUENCES[self.euler_axis_order]:
            result = quaternion_mul(axis_rotations[axis], result)
        return result

    def convert_tcb_to_bez(self) -> None:
        # TCB quaternion controls differ from BEZ ones (fillDerived_TCB 0x720910), so only euler axes convert
        if self.key_type == KeyType.EULER_KEY:
            for euler_data in self.euler_data:
                euler_data.convert_tcb_to_bez()

    def apply_time_scale(self, scale: float):
        super().apply_time_scale(scale)
        for euler_data in self.euler_data:
            euler_data.apply_time_scale(scale)
