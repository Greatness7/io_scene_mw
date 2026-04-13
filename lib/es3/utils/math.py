from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from typing import Any, TYPE_CHECKING, cast

import numpy as np
import numpy.linalg as la
from numpy import einsum, negative, sqrt, zeros as _zeros

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    Array = NDArray[Any]
    FloatArray = NDArray[np.float32]
    Float64Array = NDArray[np.float64]

ZERO = np.float32(0)

ZERO2 = np.zeros(2, dtype=np.float32)
ZERO2.setflags(write=False)

ZERO3 = np.zeros(3, dtype=np.float32)
ZERO3.setflags(write=False)

ZERO4 = np.zeros(4, dtype=np.float32)
ZERO4.setflags(write=False)

ZERO6 = np.zeros(6, dtype=np.float32)
ZERO6.setflags(write=False)

ID22 = np.eye(2, dtype=np.float32)
ID22.setflags(write=False)

ID33 = np.eye(3, dtype=np.float32)
ID33.setflags(write=False)

ID44 = np.eye(4, dtype=np.float32)
ID44.setflags(write=False)


def det(matrix: Array) -> float:
    return float(la.det(matrix))  # pyright: ignore[reportUnknownMemberType]


def allclose(left: Array, right: Array, *, rtol: float, atol: float) -> bool:
    return bool(np.allclose(left, right, rtol=rtol, atol=atol))  # pyright: ignore[reportUnknownMemberType]


def zeros(*shape: int, dtype: DTypeLike = np.float32) -> Array:
    return _zeros(shape, dtype)  # type: ignore


def _dot(left: Array, right: Array) -> Array:
    return np.dot(left, right)  # pyright: ignore[reportUnknownMemberType]


def dotproduct(arrays: list[Array]) -> Array:
    return reduce(_dot, arrays)


def compose(translation: FloatArray, rotation: FloatArray, scale: float | FloatArray) -> FloatArray:
    matrix = cast("FloatArray", ID44.copy())  # pyright: ignore[reportUnknownMemberType]
    matrix[:3, 3] = translation
    matrix[:3, :3] = rotation * scale
    return matrix


def decompose(matrix: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    translation = matrix[:3, 3]
    rotation = matrix[:3, :3]
    # scale = sqrt(square(rotation).sum(axis=0))
    scale = sqrt(einsum("ij,ij->i", rotation, rotation))
    if det(rotation) < ZERO:
        scale = negative(scale)
    translation = cast("FloatArray", translation.copy())  # pyright: ignore[reportUnknownMemberType]
    return translation, (rotation / scale), scale


def decompose_uniform(matrix: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    translation = matrix[:3, 3]
    rotation = matrix[:3, :3]
    d = float(det(rotation))
    if d < 0:
        scale = -((-d) ** (1.0 / 3.0))
    else:
        scale = d ** (1.0 / 3.0)
    translation = cast("FloatArray", translation.copy())  # pyright: ignore[reportUnknownMemberType]
    return translation, (rotation / scale), scale


def quaternion_from_euler_angle(angle: float | FloatArray, euler_axis: int, out: FloatArray | None = None) -> FloatArray:
    div2 = np.atleast_1d(angle / 2.0)  # pyright: ignore[reportUnknownMemberType]

    if out is None:
        out = zeros(len(div2), 4)
    else:
        out = np.atleast_2d(out)  # pyright: ignore[reportUnknownMemberType]

    out[:, euler_axis + 1] = 1
    out *= np.sin(div2)[:, None]
    out[:, 0] = np.cos(div2)

    return np.squeeze(out)  # pyright: ignore[reportUnknownMemberType]


def quaternion_mul(a: FloatArray, b: FloatArray, out: FloatArray | None = None) -> FloatArray:
    aw, ax, ay, zy = np.rollaxis(a, -1)
    bw, bx, by, bz = np.rollaxis(b, -1)

    result = (
        aw * bw - ax * bx - ay * by - zy * bz,
        ax * bw + aw * bx - zy * by + ay * bz,
        ay * bw + zy * bx + aw * by - ax * bz,
        zy * bw - ay * bx + ax * by + aw * bz,
    )

    return np.stack(result, axis=-1, out=out)  # pyright: ignore[reportUnknownMemberType]


def get_exact_center_radius(vertices: Array) -> tuple[FloatArray, float]:
    if len(vertices) == 0:
        return np.zeros(3, dtype=np.float32), 0.0

    from .miniball import compute as _compute  # pyright: ignore[reportUnknownVariableType]

    compute = cast("Callable[[Float64Array], tuple[Array, float]]", _compute)
    center, radius = compute(vertices.astype(dtype=np.float64, order="C", copy=False))
    center = np.array(center, dtype=np.float32)

    return center, float(radius)
