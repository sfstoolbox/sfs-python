"""Driving functions for various systems"""

import numpy as np


def pw_delay(k,x0,npw=[0, 1, 0]):
    """Plane wave by simple delaying of secondary sources"""

    d = np.exp(-1j*k*np.dot(npw,x0) )

    return d