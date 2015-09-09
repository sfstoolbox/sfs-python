"""Compute positions of various secondary source distributions.

.. plot::
    :context: reset

    import sfs
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = 8, 4  # inch
    plt.rcParams['axes.grid'] = True

"""
from __future__ import division  # for Python 2.x
import numpy as np
from . import util


def linear(N, spacing, center=[0, 0, 0], n0=[1, 0, 0]):
    """Linear secondary source distribution.

    Parameters
    ----------
    N : int
        Number of loudspeakers.
    spacing : float
        Distance (in metres) between loudspeakers.
    center : (3,) array_like, optional
        Coordinates of array center.
    n0 : (3,) array_like, optional
        Normal vector of array.

    Returns
    -------
    positions : (N, 3) numpy.ndarray
        Positions of secondary sources
    directions : (N, 3) numpy.ndarray
        Orientations (normal vectors) of secondary sources
    weights : (N,) numpy.ndarray
        Weights of secondary sources

    Example
    -------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.linear(16, 0.2, n0=[0, -1, 0])
        sfs.plot.loudspeaker_2d(x0, n0, a0)
        plt.axis('equal')

    """
    positions = np.zeros((N, 3))
    positions[:, 1] = (np.arange(N) - N/2 + 1/2) * spacing
    positions, directions = _rotate_array(positions, [1, 0, 0], [1, 0, 0], n0)
    positions += center
    directions = np.tile(directions, (N, 1))
    weights = spacing * np.ones(N)
    return positions, directions, weights


def linear_nested(N, dx1, dx2, center=[0, 0, 0], n0=[1, 0, 0]):
    """Nested linear secondary source distribution.

    Example
    -------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.linear_nested(16, 0.15, 0.2, n0=[0, -1, 0])
        sfs.plot.loudspeaker_2d(x0, n0, a0)
        plt.axis('equal')

    """
    # first segment
    x00, n00, a00 = linear(N//3, dx2, center=[0, -N//6*(dx1+dx2), 0])
    positions = x00
    directions = n00
    # second segment
    x00, n00, a00 = linear(N//3, dx1)
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    # third segment
    x00, n00, a00 = linear(N//3, dx2, center=[0, N//6*(dx1+dx2), 0])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    # compute weights
    weights = weights_linear(positions)
    # shift and rotate array
    positions, directions = _rotate_array(positions, directions, [1, 0, 0], n0)
    positions += center
    return positions, directions, weights


def linear_random(N, dy1, dy2, center=[0, 0, 0], n0=[1, 0, 0]):
    """Randomly sampled linear array.

    Example
    -------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.linear_random(12, 0.15, 0.4, n0=[0, -1, 0])
        sfs.plot.loudspeaker_2d(x0, n0, a0)
        plt.axis('equal')

    """
    # vector of uniformly distributed random distances between dy2 > dy1
    dist = dy1 + (dy2-dy1)*np.random.rand(N-1)
    # positions of secondary sources
    positions = np.zeros((N, 3))
    for m in range(1, N):
        positions[m, 1] = positions[m-1, 1] + dist[m-1]
    # weights of secondary sources
    weights = weights_linear(positions)
    # directions of secondary sources
    directions = np.tile([1, 0, 0], (N, 1))
    # shift array to origin
    positions[:, 1] -= positions[-1, 1] / 2
    # shift and rotate array
    positions, directions = _rotate_array(positions, directions, [1, 0, 0], n0)
    positions += center
    return positions, directions, weights


def circular(N, R, center=[0, 0, 0]):
    """Circular secondary source distribution parallel to the xy-plane.

    Example
    -------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.circular(16, 1)
        sfs.plot.loudspeaker_2d(x0, n0, a0, size=0.2, show_numbers=True)
        plt.axis('equal')

    """
    center = util.asarray_1d(center, dtype=np.float64)
    positions = np.tile(center, (N, 1))
    alpha = np.linspace(0, 2 * np.pi, N, endpoint=False)
    positions[:, 0] += R * np.cos(alpha)
    positions[:, 1] += R * np.sin(alpha)
    directions = np.zeros_like(positions)
    directions[:, 0] = np.cos(alpha + np.pi)
    directions[:, 1] = np.sin(alpha + np.pi)
    weights = 2 * np.pi * R / N * np.ones(N)
    return positions, directions, weights


def rectangular(Nx, dx, Ny, dy, center=[0, 0, 0], n0=None):
    """Rectangular secondary source distribution.

    Example
    -------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.rectangular(8, 0.2, 4, 0.2)
        sfs.plot.loudspeaker_2d(x0, n0, a0, show_numbers=True)
        plt.axis('equal')

    """
    # left array
    x00, n00, a00 = linear(Ny, dy)
    positions = x00
    directions = n00
    weights = a00
    # upper array
    x00, n00, a00 = linear(Nx, dx, center=[Nx/2 * dx, x00[-1, 1] + dy/2, 0],
                           n0=[0, -1, 0])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # right array
    x00, n00, a00 = linear(Ny, dy, center=[x00[-1, 0] + dx/2, 0, 0],
                           n0=[-1, 0, 0])
    x00 = np.flipud(x00)
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # lower array
    x00, n00, a00 = linear(Nx, dx, center=[Nx/2 * dx, x00[-1, 1] - dy/2, 0],
                           n0=[0, 1, 0])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # shift array to origin
    positions -= [Nx/2 * dx, 0, 0]
    # rotate array
    if n0 is not None:
        positions, directions = _rotate_array(positions, directions,
                                              [1, 0, 0], n0)
    # shift array to desired position
    positions += center
    return positions, directions, weights


def rounded_edge(Nxy, Nr, dx, center=[0, 0, 0], n0=None):
    """Array along the xy-axis with rounded edge at the origin.

    Parameters
    ----------
    Nxy : int
        Number of secondary sources along x- and y-axis.
    Nr : int
        Number of secondary sources in rounded edge.  Radius of edge is
        adjusted to equdistant sampling along entire array.
    center : (3,) array_like, optional
        Position of edge.
    n0 : (3,) array_like, optional
        Normal vector of array.  Default orientation is along xy-axis.

    Returns
    -------
    positions : (N, 3) numpy.ndarray
        Positions of secondary sources
    directions : (N, 3) numpy.ndarray
        Orientations (normal vectors) of secondary sources
    weights : (N,) numpy.ndarray
        Integration weights of secondary sources

    Example
    -------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.rounded_edge(8, 5, 0.2)
        sfs.plot.loudspeaker_2d(x0, n0, a0)
        plt.axis('equal')

    """
    # radius of rounded edge
    Nr += 1
    R = 2/np.pi * Nr * dx

    # array along y-axis
    x00, n00, a00 = linear(Nxy, dx, center=[0, Nxy//2*dx+dx/2+R, 0])
    x00 = np.flipud(x00)
    positions = x00
    directions = n00
    weights = a00

    # round part
    x00 = np.zeros((Nr, 3))
    n00 = np.zeros((Nr, 3))
    a00 = np.zeros(Nr)
    for n in range(0, Nr):
        alpha = np.pi/2 * n/Nr
        x00[n, 0] = R * (1-np.cos(alpha))
        x00[n, 1] = R * (1-np.sin(alpha))
        n00[n, 0] = np.cos(alpha)
        n00[n, 1] = np.sin(alpha)
        a00[n] = dx
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))

    # array along x-axis
    x00, n00, a00 = linear(Nxy, dx, center=[Nxy//2*dx-dx/2+R, 0, 0],
                           n0=[0, 1, 0])
    x00 = np.flipud(x00)
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))

    # rotate array
    if n0 is not None:
        positions, directions = _rotate_array(positions, directions,
                                              [1, 0, 0], n0)
    # shift array to desired position
    positions += center
    return positions, directions, weights


def planar(Ny, dy, Nz, dz, center=[0, 0, 0], n0=None):
    """Planar secondary source distribtion."""
    # initialize vectors for later np.concatenate
    positions = np.zeros((1, 3))
    directions = np.zeros((1, 3))
    for z in (np.arange(Nz) - Nz / 2 + 1 / 2) * dz:
        x00, n00, a00 = linear(Ny, dy, center=[0, 0, z])
        positions = np.concatenate((positions, x00), axis=0)
        directions = np.concatenate((directions, n00), axis=0)
    # remove first element from initialization
    positions = np.delete(positions, 0, axis=0)
    directions = np.delete(directions, 0, axis=0)
    weights = dy * dz * np.ones(Ny*Nz)
    # rotate array
    if n0 is not None:
        positions, directions = _rotate_array(positions, directions,
                                              [1, 0, 0], n0)
    # shift array to desired position
    positions += center
    return positions, directions, weights


def cube(Nx, dx, Ny, dy, Nz, dz, center=[0, 0, 0], n0=None):
    """Cube shaped secondary source distribtion."""
    # left array
    x00, n00, a00 = planar(Ny, dy, Nz, dz)
    positions = x00
    directions = n00
    weights = a00
    # upper array
    x00, n00, a00 = planar(Nx, dx, Nz, dz,
                           center=[Nx/2 * dx, x00[-1, 1] + dy/2, 0],
                           n0=[0, -1, 0])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # right array
    x00, n00, a00 = planar(Ny, dy, Nz, dz, center=[x00[-1, 0] + dx/2, 0, 0],
                           n0=[-1, 0, 0])
    x00 = np.flipud(x00)
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # lower array
    x00, n00, a00 = planar(Nx, dx, Nz, dz,
                           center=[Nx/2 * dx, x00[-1, 1] - dy/2, 0],
                           n0=[0, 1, 0])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # bottom array
    x00, n00, a00 = planar(Nx, dx, Ny, dy, center=[Nx/2 * dx, 0, -Nz/2 * dz],
                           n0=[0, 0, 1])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # top array
    x00, n00, a00 = planar(Nx, dx, Ny, dy, center=[Nx/2 * dx, 0, Nz/2 * dz],
                           n0=[0, 0, -1])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # shift array to origin
    positions -= [Nx/2 * dx, 0, 0]
    # rotate array
    if n0 is not None:
        positions, directions = _rotate_array(positions, directions,
                                              [1, 0, 0], n0)
    # shift array to desired position
    positions += center
    return positions, directions, weights


def sphere_load(fname, radius, center=[0, 0, 0]):
    """Spherical secondary source distribution loaded from datafile.

    ASCII Format (see MATLAB SFS Toolbox) with 4 numbers (3 position, 1
    weight) per secondary source located on the unit circle.

    """
    x0 = np.loadtxt(fname)
    weights = x0[:, 3]
    directions = -x0[:, :3]
    positions = center + radius * x0[:, :3]
    return positions, directions, weights


def load(fname, center=[0, 0, 0], n0=None):
    """Load secondary source positions from datafile.

    Comma Seperated Values (CSV) format with 7 values
    (3 positions, 3 directions, 1 weight) per secondary source.

    """
    data = np.loadtxt(fname, delimiter=',')
    positions = data[:, [0, 1, 2]]
    directions = data[:, [3, 4, 5]]
    weights = np.squeeze(data[:, [6]])
    # rotate array
    if n0 is not None:
        positions, directions = _rotate_array(positions, directions,
                                              [1, 0, 0], n0)
    # shift array to desired position
    positions += center
    return positions, directions, weights


def weights_linear(positions):
    """Calculate loudspeaker weights for a linear array.

    The linear array has to be parallel to the y-axis.

    """
    N = len(positions)
    weights = np.zeros(N)
    dy = np.diff(positions[:, 1])
    weights[0] = dy[0]
    for m in range(1, N - 1):
        weights[m] = 0.5 * (dy[m-1] + dy[m])
    weights[-1] = dy[-1]
    return np.abs(weights)


def weights_closed(positions):
    """Calculate loudspeaker weights for a simply connected array.

    The weights are calculated according to the midpoint rule.

    Note: The loudspeaker positions have to be ordered on the closed
    contour.

    """
    positions = util.asarray_of_rows(positions)
    if len(positions) == 0:
        weights = []
    elif len(positions) == 1:
        weights = [1.0]
    else:
        successors = np.roll(positions, -1, axis=0)
        d = [np.linalg.norm(b - a) for a, b in zip(positions, successors)]
        weights = [0.5 * (a + b) for a, b in zip(d, d[-1:] + d)]
    return np.array(weights)


def _rotate_array(x0, n0, n1, n2):
    """Rotate secondary sources from n1 to n2."""
    R = util.rotation_matrix(n1, n2)
    x0 = np.inner(x0, R)
    n0 = np.inner(n0, R)
    return x0, n0
