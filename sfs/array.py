"""Compute positions of various secondary source distributions."""

import numpy as np
from . import util


def linear(N, dx, center=[0, 0, 0], n0=None):
    """Linear secondary source distribution."""
    center = np.squeeze(np.asarray(center, dtype=np.float64))
    positions = np.zeros((N, 3))
    positions[:, 1] = (np.arange(N) - N / 2 + 1 / 2) * dx
    if n0 is None:
        n0 = np.array([1, 0, 0], dtype=np.float64)
    else:
        n0 = np.array(n0, dtype=np.float64)
        R = util.rotation_matrix([1, 0, 0], n0)
        positions = np.inner(positions, R)
    positions += center
    directions = np.tile(n0, (N, 1))
    return positions, directions


def circular(N, R, center=[0, 0, 0]):
    """Circular secondary source distribution parallel to the xy-plane."""
    center = np.squeeze(np.asarray(center, dtype=np.float64))
    positions = np.tile(center, (N, 1))
    alpha = np.linspace(0, 2 * np.pi, N, endpoint=False)
    positions[:, 0] += R * np.cos(alpha)
    positions[:, 1] += R * np.sin(alpha)
    directions = np.zeros_like(positions)
    directions[:, 0] = np.cos(alpha + np.pi)
    directions[:, 1] = np.sin(alpha + np.pi)
    return positions, directions


def rectangular(Nx, dx, Ny, dy, center=[0, 0, 0]):
    """Rectangular secondary source distribution."""

    # left array
    x00, n00 = linear(Ny, dy)
    positions = x00
    directions = n00
    # upper array
    x00, n00 = linear(Nx, dx, center=[Nx/2 * dx, x00[-1, 1] + dy/2, 0],
                      n0=[0, -1, 0])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    # right array
    x00, n00 = linear(Ny, dy, center=[x00[-1, 0] + dx/2, 0, 0], n0=[-1, 0, 0])
    x00 = np.flipud(x00)
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    # lower array
    x00, n00 = linear(Nx, dx, center=[Nx/2 * dx, x00[-1, 1] - dy/2, 0],
                      n0=[0, 1, 0])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))

    positions += np.asarray(center) - np.asarray([Nx/2 * dx, 0, 0])

    return positions, directions
