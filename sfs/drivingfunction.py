"""Driving functions for various systems"""

import numpy as np


def pw_delay(k, x0, npw=[0, 1, 0]):
    """Plane wave by simple delaying of secondary sources."""
    x0 = np.asarray(x0)
    npw = np.squeeze(np.asarray(npw))
    return np.exp(-1j * k * np.inner(npw, x0))
