# NumPy's overloads read as partially unknown under strict mode; the arrays here are typed explicitly.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, cast

from ..utils.math import np, zeros
from ..utils.typing import *

from .NiObject import NiObject

if TYPE_CHECKING:
    from ..utils.math import Array
    from .NiBinaryStream import NiBinaryStream


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

    def load(self, stream: NiBinaryStream) -> None:
        num_keys = stream.read_uint()
        if num_keys:
            self.key_type = KeyType(stream.read_int())
            self.keys = stream.read_floats(num_keys, self.key_size)

    def save(self, stream: NiBinaryStream) -> None:
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

    def get_start_stop_times(self) -> tuple[float, float]:
        if len(self.keys) == 0:
            return (0, 0)
        else:
            return (self.keys[0, 0], self.keys[-1, 0])

    def get_tangent_handles(self):
        """Bezier handles for BEZ keys. TCB keys are first converted to BEZ in place (exactly)."""
        if self.key_type == KeyType.TCB_KEY:
            self.convert_tcb_to_bez()
        if self.key_type == KeyType.BEZ_KEY:
            return self.get_bez_tangent_handles()

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
            in_dy = cast("Array", self.in_tans.T) / 3.0
            out_dy = cast("Array", self.out_tans.T) / 3.0

            # incoming handles
            handles[0, ..., 0] = times - in_dx
            handles[0, ..., 1] = values.T - in_dy
            # outgoing handles
            handles[1, ..., 0] = times + out_dx
            handles[1, ..., 1] = values.T + out_dy

        return handles

    # Key evaluation follows Morrowind.exe: BEZ and TCB segments are cubic Hermite curves whose tangents are
    # in "value change per whole segment" units (sg_interpolate_hermite 0x7266a0), and times outside the
    # key range hold the first/last value (genInterp 0x70cfe0). Work stays in the keys' own dtype (float32),
    # the precision the engine plays them at.

    def _columns(self) -> tuple[Array, Array]:
        """Times (n,) and values (n, m), as views of the keys where possible."""
        times = self.times
        return times, self.values.reshape(len(times), -1)

    def _tangents(self, times: Array, values: Array) -> tuple[Array, Array]:
        """In and out tangents (n, m), derived from T/C/B for TCB keys."""
        if self.key_type == KeyType.TCB_KEY:
            return self._tcb_tangents(times, values, self.tcb)
        return self.in_tans.reshape(values.shape), self.out_tans.reshape(values.shape)

    @staticmethod
    def _tcb_tangents(times: Array, values: Array, tcb: Array) -> tuple[Array, Array]:
        """Kochanek-Bartels tangents as derived at load (NiFloatKey fillDerived_TCB 0x720df0, NiPosKey 0x71fdf0).

        End keys use virtual neighbours v[-1] = 2v[0] - v[1] and v[n] = 2v[n-1] - v[n-2], and unit spacing.
        """
        n = len(times)
        if n < 2:
            return np.zeros_like(values), np.zeros_like(values)

        deltas = np.diff(values, axis=0)
        vd0 = np.concatenate((deltas[:1], deltas))
        vd1 = np.concatenate((deltas, deltas[-1:]))

        spacing = np.diff(times)
        dt0, dt1 = np.ones_like(times), np.ones_like(times)  # end keys keep unit spacing
        dt0[1:-1] = spacing[:-1]
        dt1[1:-1] = spacing[1:]

        tension, continuity, bias = (tcb[:, i, None] for i in range(3))
        a = 0.5 * (1.0 - tension) * (1.0 + continuity)
        b = 0.5 * (1.0 - tension) * (1.0 - continuity)

        total = dt0 + dt1
        in_scale = np.divide(2.0 * dt0, total, out=np.zeros_like(total), where=total != 0)[:, None]
        out_scale = np.divide(2.0 * dt1, total, out=np.zeros_like(total), where=total != 0)[:, None]

        in_tans = (a * (1.0 - bias) * vd1 + b * (1.0 + bias) * vd0) * in_scale
        out_tans = (b * (1.0 - bias) * vd1 + a * (1.0 + bias) * vd0) * out_scale
        return in_tans, out_tans

    @staticmethod
    def _hermite(f: Array, p0: Array, m0: Array, p1: Array, m1: Array) -> Array:
        """Cubic Hermite value at fractions `f` (k, 1)."""
        f2 = f * f
        f3 = f2 * f
        return (2 * f3 - 3 * f2 + 1) * p0 + (f3 - 2 * f2 + f) * m0 + (3 * f2 - 2 * f3) * p1 + (f3 - f2) * m1

    @staticmethod
    def _hermite_slope(f: Array, p0: Array, m0: Array, p1: Array, m1: Array) -> Array:
        """Derivative d/df of `_hermite`."""
        f2 = f * f
        return (6 * f2 - 6 * f) * (p0 - p1) + (3 * f2 - 4 * f + 1) * m0 + (3 * f2 - 2 * f) * m1

    @staticmethod
    def _locate(times: Array, query: Array) -> tuple[Array, Array]:
        """Segment index and fraction in [0, 1] of each query time; fractions clamp outside the key range."""
        segment = np.clip(np.searchsorted(times, query, side="right") - 1, 0, len(times) - 2)
        start, stop = times[segment], times[segment + 1]
        duration = stop - start
        fraction = np.divide(query - start, duration, out=np.zeros_like(query), where=duration > 0)
        fraction[query >= stop] = 1.0
        return segment, np.clip(fraction, 0.0, 1.0)

    def evaluate(self, times: Array) -> Array:
        """Values at `times`, shaped (len(times), m). Times outside the key range hold the end values."""
        query = np.atleast_1d(np.asarray(times, dtype=self.keys.dtype))
        key_times, values = self._columns()
        n = len(key_times)
        if n == 0:
            raise ValueError(f"{self.type} has no keys to evaluate")
        if n == 1:
            return np.repeat(values, len(query), axis=0)

        segment, fraction = self._locate(key_times, query)
        f = fraction[:, None]
        p0, p1 = values[segment], values[segment + 1]
        if self.key_type == KeyType.LIN_KEY:
            return p0 + f * (p1 - p0)

        in_tans, out_tans = self._tangents(key_times, values)
        return self._hermite(f, p0, out_tans[segment], p1, in_tans[segment + 1])

    def evaluate_and_insert(self, times: Array, atol: float = 1e-6) -> None:
        """Insert keys at `times` without changing the curve. Times within `atol` of a key are skipped.

        BEZ segments are split exactly; TCB keys are converted to BEZ first. A key inserted outside the key
        range holds the end value with flat tangents, which is what the engine plays there.
        """
        if len(self.keys) == 0:
            raise ValueError(f"{self.type} has no keys to evaluate")

        key_times = self.times
        n = len(key_times)

        query = np.unique(np.asarray(times, dtype=self.keys.dtype))
        nearest = np.searchsorted(key_times, query)
        below = np.abs(query - key_times[np.clip(nearest - 1, 0, n - 1)])
        above = np.abs(key_times[np.clip(nearest, 0, n - 1)] - query)
        query = query[np.minimum(below, above) > atol]
        if len(query) == 0:
            return
        query = query[np.concatenate(((True,), np.diff(query) > atol))]

        if self.key_type == KeyType.TCB_KEY:
            self.convert_tcb_to_bez()
        key_times, values = self._columns()

        # the merged table is written once, in place: each key's row is its index plus the number of keys
        # from the other list that precede it (both lists are sorted and share no times)
        rows = np.arange(n) + np.searchsorted(query, key_times)
        new_rows = np.arange(len(query)) + np.searchsorted(key_times, query)
        keys = np.empty((n + len(query), self.key_size), self.keys.dtype)
        keys[rows, 0] = key_times
        keys[new_rows, 0] = query

        m = values.shape[1]
        merged_values = keys[:, 1 : 1 + m]
        merged_values[rows] = values
        if self.key_type == KeyType.LIN_KEY:
            merged_values[new_rows] = self.evaluate(query)
            self.keys = keys
            return

        in_tans, out_tans = self._tangents(key_times, values)
        merged_in = keys[:, 1 + m : 1 + 2 * m]
        merged_out = keys[:, 1 + 2 * m :]

        # tangents are first written as slopes (value per time), then scaled by the merged spacing
        spacing = np.diff(key_times)
        inverse = np.divide(1.0, spacing, out=np.zeros_like(spacing), where=spacing > 0)[:, None]
        merged_in[rows[0]] = 0.0
        merged_in[rows[1:]] = in_tans[1:] * inverse
        merged_out[rows[:-1]] = out_tans[:-1] * inverse
        merged_out[rows[-1]] = 0.0

        # inserted keys: the curve's value and slope, flat outside the key range
        if n == 1:
            merged_values[new_rows] = values[0]
            merged_in[new_rows] = merged_out[new_rows] = 0.0
        else:
            segment, fraction = self._locate(key_times, query)
            segment_keys = (fraction[:, None], values[segment], out_tans[segment], values[segment + 1], in_tans[segment + 1])
            merged_values[new_rows] = self._hermite(*segment_keys)
            slope = self._hermite_slope(*segment_keys)
            slope *= inverse[segment]
            slope[(query < key_times[0]) | (query > key_times[-1])] = 0.0
            merged_in[new_rows] = merged_out[new_rows] = slope

        merged_spacing = np.diff(keys[:, 0])[:, None]
        merged_in[1:] *= merged_spacing
        merged_out[:-1] *= merged_spacing
        # an original end key that stays at the end keeps the tangent it faces outward with
        if rows[0] == 0:
            merged_in[0] = in_tans[0]
        if rows[-1] == len(keys) - 1:
            merged_out[-1] = out_tans[-1]

        self.keys = keys

    def convert_tcb_to_bez(self) -> None:
        """Replace TCB keys with the BEZ keys the engine derives from them. The curve is unchanged."""
        if self.key_type != KeyType.TCB_KEY:
            return
        times, values = self._columns()
        in_tans, out_tans = self._tangents(times, values)
        self.key_type = KeyType.BEZ_KEY
        self.keys = np.column_stack((times, values, in_tans, out_tans))

    def apply_time_scale(self, scale: float):
        super().apply_time_scale(scale)
        if len(self.keys):
            self.times[:] *= scale
