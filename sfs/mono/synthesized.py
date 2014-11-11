"""Computation of synthesized sound fields."""

import sfs
import numpy as np


def generic(omega, x0, d, x, y, z, c=None):
    """Compute sound field for a generic driving function"""
    d = np.squeeze(np.asarray(d))
    if len(d) != len(x0):
        raise ValueError("length mismatch")
    p = 0
    for weight, position in zip(d, x0):
        p += weight * sfs.mono.source.point(omega, position, x, y, z, c)
    return p
