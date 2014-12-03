"""Plot sound fields etc"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from . import util


def loudspeaker_2d(x0, n0, a0=None, w=0.08, h=0.08):
    """Draw loudspeaker symbols at given locations, angles"""
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    patches = []
    fc = []
    if a0 is None:
        a0 = 0.5 * np.ones(len(x0))
    else:
        a0 = np.asarray(a0)

    # coordinates of loudspeaker symbol
    v01 = np.asarray([[-h, -h, -h / 2, -h / 2, -h], [-w / 2, w / 2, w / 2,
                      -w / 2, -w / 2], [0, 0, 0, 0, 0]])
    v02 = np.asarray(
        [[-h / 2, 0, 0, -h / 2], [-w / 6, -w / 2, w / 2, w / 6], [0, 0, 0, 0]])

    v01 = v01.T
    v02 = v02.T

    for x00, n00, a00 in zip(x0, n0, a0):
        # rotate and translate coordinates
        R = util.rotation_matrix([1, 0, 0], n00)
        v1 = np.inner(v01, R) + x00
        v2 = np.inner(v02, R) + x00

        # add coordinates to list of patches
        polygon = Polygon(v1[:, :-1], True)
        patches.append(polygon)
        polygon = Polygon(v2[:, :-1], True)
        patches.append(polygon)

        # set facecolor (two times due to split patches)
        fc.append((1-a00) * np.ones(3))
        fc.append((1-a00) * np.ones(3))

    # add collection of patches to current axis
    p = PatchCollection(patches, edgecolor='0', facecolor=fc, alpha=1)
    ax = plt.gca()
    ax.add_collection(p)


def loudspeaker_3d(x0, n0, a0=None, w=0.08, h=0.08):
    """Plot positions and normal vectors of a 3D secondary source
    distribution."""
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x0[:, 0], x0[:, 1], x0[:, 2], n0[:, 0],
              n0[:, 1], n0[:, 2], length=0.1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Secondary Sources')
    fig.show()


def soundfield(p, x, y, xnorm=[0, 0, 0]):
    """Two-dimensional plot of sound field"""

    # normalize sound field wrt xnorm
    xx, yy = np.meshgrid(x - xnorm[0], y - xnorm[1], sparse=True)
    r = np.sqrt((xx) ** 2 + (yy) ** 2)
    idx = np.unravel_index(r.argmin(), r.shape)
    p = p / abs(p[idx])

    # plot sound field
    plt.imshow(np.real(p), cmap=plt.cm.RdBu, origin='lower',
               extent=[min(x), max(x), min(y), max(y)], vmax=2, vmin=-2,
               aspect='equal')

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar()
