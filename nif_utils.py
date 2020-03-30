import types
import numpy as np
import copy as _copy


def dispatcher(func):
    registry = {}

    def register(name, func=None):
        if func is None:
            return lambda x: register(name, x)
        return registry.setdefault(name, func)

    def wrapper(self, node):
        name = getattr(node.source, "type", "NONE")
        return registry.get(name, func)(self, node)

    wrapper.register = register

    return wrapper


class Namespace(types.SimpleNamespace):
    from copy import copy, deepcopy


# --------------
# MATH FUNCTIONS
# --------------

def unique_rows(*arrays, precision=0.001):
    targets = list(filter(len, arrays))
    rounded = np.round(np.column_stack(targets) / precision) * precision
    _, idx, inv = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    return idx, inv


def quaternion_from_matrix(m):
    ((m00, m10, m20),
     (m01, m11, m21),
     (m02, m12, m22)) = m[:3, :3].conj().tolist()

    if m22 < 0:
        if m00 > m11:
            t = 1 + m00 - m11 - m22
            q = (m12 - m21), t, (m01 + m10), (m20 + m02)
        else:
            t = 1 - m00 + m11 - m22
            q = (m20 - m02), (m01 + m10), t, (m12 + m21)
    else:
        if m00 < -m11:
            t = 1 - m00 - m11 + m22
            q = (m01 - m10), (m20 + m02), (m12 + m21), t
        else:
            t = 1 + m00 + m11 + m22
            q = t, (m12 - m21), (m20 - m02), (m01 - m10)

    return np.fromiter(q, m.dtype, 4) * 0.5 / np.sqrt(t)


def quaternion_mul(a, b, out=None):
    aw, ax, ay, zy = np.rollaxis(a, -1, 0)
    bw, bx, by, bz = np.rollaxis(b, -1, 0)

    result = (
        aw * bw - ax * bx - ay * by - zy * bz,
        ax * bw + aw * bx - zy * by + ay * bz,
        ay * bw + zy * bx + aw * by - ax * bz,
        zy * bw - ay * bx + ax * by + aw * bz,
    )

    return np.stack(result, axis=-1, out=out)


def snap_rotation(m):
    m_abs = np.abs(m)
    m_out = np.zeros_like(m)

    for row in m:
        x, y = np.unravel_index(m_abs.argmax(), m.shape)
        m_out[x, y] = -1 if m[x, y] < 0 else 1
        m_abs[x, :] = m_abs[:, y] = 0

    return m_out
