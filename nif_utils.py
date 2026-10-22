import numpy as np

# --------------
# MATH FUNCTIONS
# --------------

def snap_rotation(m):
    m_abs = np.abs(m)
    m_out = np.zeros_like(m)

    for row in m:
        x, y = np.unravel_index(m_abs.argmax(), m.shape)
        m_out[x, y] = -1 if m[x, y] < 0 else 1
        m_abs[x, :] = m_abs[:, y] = 0

    return m_out
