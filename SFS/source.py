import numpy as np


def point(k, position, x, y):

    xx, yy = np.meshgrid(x-position[0], y-position[1], sparse=True)
    r = np.sqrt((xx)**2 + (yy)**2)
    z = np.exp(-1j*k*r)/r

    return z

