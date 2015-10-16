"""Computation of synthesized sound fields."""

import numpy as np
from .source import point


def generic(omega, x0, n0, d, grid, c=None, source=point):
    """Compute sound field for a generic driving function."""
    d = np.squeeze(np.asarray(d))
    if len(d) != len(x0):
        raise ValueError("length mismatch")
    p = 0
    for weight, position, direction in zip(d, x0, n0):
        if weight != 0:
            p += weight * source(omega, position, direction, grid, c)
    return p


def shiftphase(p, phase):
    """Shift phase of a sound field."""
    p = np.asarray(p)
    return p * np.exp(1j * phase)
