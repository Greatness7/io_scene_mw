from __future__ import annotations

from functools import reduce

import numpy as np
import numpy.linalg as la
from numpy import dot, einsum, negative, sqrt, zeros as _zeros

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

det = la._umath_linalg.det

if __name__ == "__main__":
    from es3.utils.typing import *


def zeros(*shape: int, dtype=np.float32) -> ndarray:
    return _zeros(shape, dtype)  # type: ignore


def dotproduct(arrays: list[ndarray]) -> ndarray:
    return reduce(dot, arrays)


def compose(translation: ndarray, rotation: ndarray, scale: float |  ndarray) -> ndarray:
    matrix = ID44.copy()
    matrix[:3, 3] = translation
    matrix[:3, :3] = rotation * scale
    return matrix


def decompose(matrix: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    translation = matrix[:3, 3]
    rotation = matrix[:3, :3]
    # scale = sqrt(square(rotation).sum(axis=0))
    scale = sqrt(einsum("ij,ij->i", rotation, rotation))
    if det(rotation) < ZERO:
        scale = negative(scale)
    return translation.copy(), (rotation / scale), scale


def decompose_uniform(matrix: ndarray) -> tuple[ndarray, ndarray, float]:
    translation = matrix[:3, 3]
    rotation = matrix[:3, :3]
    d = float(det(rotation))
    if d < 0:
        scale = -(-d) ** (1.0 / 3.0)
    else:
        scale = d ** (1.0 / 3.0)
    return translation.copy(), (rotation / scale), scale


def quaternion_from_euler_angle(angle: float |  ndarray, euler_axis: int, out=None) -> ndarray:
    div2 = np.atleast_1d(angle / 2.0)

    if out is None:
        out = zeros(len(div2), 4)
    else:
        out = np.atleast_2d(out)

    out[:, euler_axis + 1] = 1
    out *= np.sin(div2)[:, None]
    out[:, 0] = np.cos(div2)

    return np.squeeze(out)


def quaternion_mul(a, b, out=None):
    aw, ax, ay, zy = np.rollaxis(a, -1)
    bw, bx, by, bz = np.rollaxis(b, -1)

    result = (
        aw * bw - ax * bx - ay * by - zy * bz,
        ax * bw + aw * bx - zy * by + ay * bz,
        ay * bw + zy * bx + aw * by - ax * bz,
        zy * bw - ay * bx + ax * by + aw * bz,
    )

    return np.stack(result, axis=-1, out=out)
