"""3D plots of sound fields etc."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def secondary_sources(x0, n0, a0=None, *, w=0.08, h=0.08):
    """Plot positions and normals of a 3D secondary source distribution."""
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    q = ax.quiver(x0[:, 0], x0[:, 1], x0[:, 2], n0[:, 0],
                  n0[:, 1], n0[:, 2], length=0.1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Secondary Sources')
    return q
