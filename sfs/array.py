"""Positions of various secondary source distributions"""

import numpy as np


def linear(N, dx, center=[0, 0, 0]):
    """Linear secondary source distribution parallel to the x-axis."""
    xpos = (np.arange(N) - N/2 + 1/2) * dx
    center = np.squeeze(np.asarray(center, dtype=np.float64))
    positions = np.tile(center, (N, 1))
    positions[:, 0] += xpos
    return positions
