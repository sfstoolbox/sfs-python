"""Compute positions of various secondary source distributions."""

import numpy as np


def linear(N, dx, center=[0, 0, 0]):
    """Linear secondary source distribution parallel to the x-axis."""
    xpos = (np.arange(N) - N / 2 + 1 / 2) * dx
    center = np.squeeze(np.asarray(center, dtype=np.float64))
    positions = np.tile(center, (N, 1))
    positions[:, 0] += xpos
    directions = np.tile(np.array([0, 1, 0], dtype=np.float64), (N, 1))
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
