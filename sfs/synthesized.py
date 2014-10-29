"""Computation of synthesized sound field"""

import sfs
import numpy as np

def generic(x, y, z, x0, k, d, twin):
    """Compute sound field for a generic driving function"""
    d = np.squeeze(np.asarray(d))
    twin = np.squeeze(np.asarray(twin))
    weights = d * twin
    if len(weights) != len(x0):
        raise ValueError("length mismatch")
    p = 0
    for weight, position in zip(weights, x0):
        p += weight * sfs.source.point(k, position, x, y, z)
    return p
