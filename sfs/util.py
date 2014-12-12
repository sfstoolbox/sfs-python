"""Various utility functions"""

import numpy as np
from . import defs


def rotation_matrix(n1, n2):
    """Compute rotation matrix for rotation from n1 to n2"""
    n1 = np.asarray(n1)
    n2 = np.asarray(n2)
    # no rotation required
    if all(n1 == n2):
        return np.eye(3)

    v = np.cross(n1, n2)
    s = np.linalg.norm(v)

    # check for rotation of 180deg around one axis
    if s == 0:
        rot = np.identity(3)
        for i in np.arange(3):
            if np.abs(n1[i]) > 0 and np.abs(n1[i]) > 0 and n1[i] == -n2[i]:
                rot[i, i] = -1
        return rot

    c = np.inner(n1, n2)
    vx = [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]

    return np.identity(3) + vx + np.dot(vx, vx) * (1 - c) / s ** 2


def wavenumber(omega, c=None):
    """Compute the wavenumber for a given radial frequency"""
    if c is None:
        c = defs.c
    return omega / c


def normal(alpha, beta):
    """Compute normal vector from azimuth, colatitude."""
    return [np.cos(alpha) * np.sin(beta), np.sin(alpha) * np.sin(beta),
            np.cos(beta)]


def sph2cart(alpha, beta, r):
    """Spherical to cartesian coordinates."""
    x = r * np.cos(alpha) * np.sin(beta)
    y = r * np.sin(alpha) * np.sin(beta)
    z = r * np.cos(beta)

    return x, y, z


def cart2sph(x, y, z):
    """Cartesian to spherical coordinates."""
    alpha = np.arctan2(y, x)
    beta = np.arccos(z / np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)

    return alpha, beta, r
