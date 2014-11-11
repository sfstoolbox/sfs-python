"""Various utility functions"""

import numpy as np
from . import defs


def rotation_matrix(n1, n2):
    """Compute rotation matrix for rotation from n1 to n2"""
    n1 = np.asarray(n1)
    n2 = np.asarray(n2)

    v = np.cross(n1, n2)
    s = np.linalg.norm(v)
    c = np.inner(n1, n2)
    vx = [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]

    return np.identity(3) + vx + np.dot(vx, vx) * (1 - c) / s ** 2
    

def wavenumber(omega, c=None):
    """Compute the wavenumber for a given radial frequency"""
    if c is None:
        c = defs.c
    return omega / c