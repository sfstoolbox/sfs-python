"""Plot sound fields etc"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from . import util


def loudspeaker(x0, n0, w=0.08, h=0.08):
    """Draw loudspeaker symbols at given locations, angles"""
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    patches = []

    # coordinates of loudspeaker symbol
    v01 = np.asarray([[-h, -h, -h / 2, -h /2, -h], [-w / 2, w / 2, w / 2, 
                      -w / 2, -w / 2], [0, 0, 0, 0, 0]])
    v02 = np.asarray(
        [[-h / 2, 0, 0, -h / 2], [-w / 6, -w / 2, w / 2, w / 6], [0, 0, 0, 0]])
    v01 = v01.T
    v02 = v02.T

    for x00, n00 in zip(x0, n0):
        # rotate and translate coordinates
        R = util.rotation_matrix([1, 0, 0], n00)
        v1 = np.inner(v01, R) + x00
        v2 = np.inner(v02, R) + x00

        # add coordinates to list of patches
        polygon = Polygon(v1[:, :-1], True)
        patches.append(polygon)
        polygon = Polygon(v2[:, :-1], True)
        patches.append(polygon)

    # add collection of patches to current axis
    p = PatchCollection(patches, edgecolor='0', facecolor='.5', alpha=1)
    ax = plt.gca()
    ax.add_collection(p)
