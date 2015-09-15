"""Compute positions of various secondary source distributions.

.. plot::
    :context: reset

    import sfs
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = 8, 4  # inch
    plt.rcParams['axes.grid'] = True

.. autoclass:: ArrayData

"""
from __future__ import division  # for Python 2.x
from collections import namedtuple
import numpy as np
from . import util


ArrayData = namedtuple('ArrayData', 'x n a')
"""Named tuple returned by array functions.

Attributes
----------
x : (N, 3) numpy.ndarray
    Positions of secondary sources
n : (N, 3) numpy.ndarray
    Orientations (normal vectors) of secondary sources
a : (N,) numpy.ndarray
    Weights of secondary sources

"""


def linear(N, spacing, center=[0, 0, 0], n0=[1, 0, 0]):
    """Linear secondary source distribution.

    Parameters
    ----------
    N : int
        Number of secondary sources.
    spacing : float
        Distance (in metres) between secondary sources.
    center : (3,) array_like, optional
        Coordinates of array center.
    n0 : (3,) array_like, optional
        Normal vector of array.

    Returns
    -------
    ArrayData
        Positions, orientations and weights of secondary sources.
        See :class:`ArrayData`.

    Example
    -------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.linear(16, 0.2, n0=[0, -1, 0])
        sfs.plot.loudspeaker_2d(x0, n0, a0)
        plt.axis('equal')

    """
    return _linear_helper(np.arange(N) * spacing, center, n0)


def linear_diff(distances, center=[0, 0, 0], n0=[1, 0, 0]):
    """Linear secondary source distribution from a list of distances.

    Parameters
    ----------
    distances : (N-1,) array_like
        Sequence of secondary sources distances in metres.
    center, n0
        See :func:`linear`

    Returns
    -------
    ArrayData
        Positions, orientations and weights of secondary sources.
        See :class:`ArrayData`.

    Example
    -------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.linear_diff(4 * [0.3] + 6 * [0.15] + 4 * [0.3], n0=[0, -1, 0])
        sfs.plot.loudspeaker_2d(x0, n0, a0)
        plt.axis('equal')

    """
    distances = util.asarray_1d(distances)
    ycoordinates = np.concatenate(([0], np.cumsum(distances)))
    return _linear_helper(ycoordinates, center, n0)


def linear_random(N, min_spacing, max_spacing, center=[0, 0, 0],
                  n0=[1, 0, 0], seed=None):
    """Randomly sampled linear array.

    Parameters
    ----------
    N : int
        Number of secondary sources.
    min_spacing, max_spacing : float
        Minimal and maximal distance (in metres) between secondary
        sources.
    center, n0
        See :func:`linear`
    seed : {None, int, array_like}, optional
        Random seed.  See :class:`numpy.random.RandomState`.

    Returns
    -------
    ArrayData
        Positions, orientations and weights of secondary sources.
        See :class:`ArrayData`.

    Example
    -------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.linear_random(12, 0.15, 0.4, n0=[0, -1, 0])
        sfs.plot.loudspeaker_2d(x0, n0, a0)
        plt.axis('equal')

    """
    r = np.random.RandomState(seed)
    distances = r.uniform(min_spacing, max_spacing, size=N-1)
    return linear_diff(distances, center, n0)


def circular(N, R, center=[0, 0, 0]):
    """Circular secondary source distribution parallel to the xy-plane.

    Parameters
    ----------
    N : int
        Number of secondary sources.
    R : float
        Radius in metres.
    center
        See :func:`linear`.

    Returns
    -------
    ArrayData
        Positions, orientations and weights of secondary sources.
        See :class:`ArrayData`.

    Example
    -------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.circular(16, 1)
        sfs.plot.loudspeaker_2d(x0, n0, a0, size=0.2, show_numbers=True)
        plt.axis('equal')

    """
    center = util.asarray_1d(center)
    alpha = np.linspace(0, 2 * np.pi, N, endpoint=False)
    positions = np.zeros((N, len(center)))
    positions[:, 0] = R * np.cos(alpha)
    positions[:, 1] = R * np.sin(alpha)
    positions += center
    normals = np.zeros_like(positions)
    normals[:, 0] = np.cos(alpha + np.pi)
    normals[:, 1] = np.sin(alpha + np.pi)
    weights = np.ones(N) * 2 * np.pi * R / N
    return ArrayData(positions, normals, weights)


def rectangular(N, spacing, center=[0, 0, 0], n0=[1, 0, 0]):
    """Rectangular secondary source distribution.

    Parameters
    ----------
    N : int or pair of int
        Number of secondary sources on each side of the rectangle.
        If a pair of numbers is given, the first one specifies the first
        and third segment, the second number specifies the second and
        fourth segment.
    spacing : float or pair of float
        Distance (in metres) between secondary sources.
        If a pair of numbers is given, the first one specifies the first
        and third segment, the second number specifies the second and
        fourth segment.
    center, n0
        See :func:`linear`.  The normal vector `n0` corresponds to the
        first linear segment.

    Returns
    -------
    ArrayData
        Positions, orientations and weights of secondary sources.
        See :class:`ArrayData`.

    Example
    -------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.rectangular((4, 8), 0.2)
        sfs.plot.loudspeaker_2d(x0, n0, a0, show_numbers=True)
        plt.axis('equal')

    """
    N1, N2 = (N, N) if np.isscalar(N) else N
    d1, d2 = (spacing, spacing) if np.isscalar(spacing) else spacing
    offset1 = N2/2 * d2
    offset2 = N1/2 * d1
    positions, normals, weights = concatenate(
        linear(N1, d1, center=[-offset1, 0, 0], n0=[1, 0, 0]),  # left
        linear(N2, d2, center=[0, offset2, 0], n0=[0, -1, 0]),  # upper
        linear(N1, d1, center=[offset1, 0, 0], n0=[-1, 0, 0]),  # right
        linear(N2, d2, center=[0, -offset2, 0], n0=[0, 1, 0]),  # lower
    )
    positions, normals = _rotate_array(positions, normals, [1, 0, 0], n0)
    positions += center
    return ArrayData(positions, normals, weights)


def rounded_edge(Nxy, Nr, dx, center=[0, 0, 0], n0=[1, 0, 0]):
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
    ArrayData
        Positions, orientations and weights of secondary sources.
        See :class:`ArrayData`.

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
    positions, directions = _rotate_array(positions, directions, [1, 0, 0], n0)
    # shift array to desired position
    positions += center
    return ArrayData(positions, directions, weights)


def planar(Ny, dy, Nz, dz, center=[0, 0, 0], n0=[1, 0, 0]):
    """Planar secondary source distribtion.

    Returns
    -------
    ArrayData
        Positions, orientations and weights of secondary sources.
        See :class:`ArrayData`.

    """
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
    positions, directions = _rotate_array(positions, directions, [1, 0, 0], n0)
    # shift array to desired position
    positions += center
    return ArrayData(positions, directions, weights)


def cube(Nx, dx, Ny, dy, Nz, dz, center=[0, 0, 0], n0=[1, 0, 0]):
    """Cube shaped secondary source distribtion.

    Returns
    -------
    ArrayData
        Positions, orientations and weights of secondary sources.
        See :class:`ArrayData`.

    """
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
    positions, directions = _rotate_array(positions, directions, [1, 0, 0], n0)
    # shift array to desired position
    positions += center
    return ArrayData(positions, directions, weights)


def sphere_load(fname, radius, center=[0, 0, 0]):
    """Spherical secondary source distribution loaded from datafile.

    ASCII Format (see MATLAB SFS Toolbox) with 4 numbers (3 position, 1
    weight) per secondary source located on the unit circle.

    Returns
    -------
    ArrayData
        Positions, orientations and weights of secondary sources.
        See :class:`ArrayData`.

    """
    data = np.loadtxt(fname)
    positions, weights = data[:, :3], data[:, 3]
    normals = -positions
    positions *= radius
    positions += center
    return ArrayData(positions, normals, weights)


def load(fname, center=[0, 0, 0], n0=[1, 0, 0]):
    """Load secondary source positions from datafile.

    Comma Seperated Values (CSV) format with 7 values
    (3 positions, 3 normal vectors, 1 weight) per secondary source.

    Returns
    -------
    ArrayData
        Positions, orientations and weights of secondary sources.
        See :class:`ArrayData`.

    """
    data = np.loadtxt(fname, delimiter=',')
    positions, normals, weights = data[:, :3], data[:, 3:6], data[:, 6]
    positions, normals = _rotate_array(positions, normals, [1, 0, 0], n0)
    positions += center
    return ArrayData(positions, normals, weights)


def weights_linear(positions):
    """Calculate loudspeaker weights for a linear array."""
    positions = util.asarray_of_rows(positions)
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.array([distances[0]] +
                    [np.mean(pair) for pair in zip(distances, distances[1:])] +
                    [distances[-1]])


def weights_closed(positions):
    """Calculate loudspeaker weights for a simply connected array.

    The weights are calculated according to the midpoint rule.

    Note: The loudspeaker positions have to be ordered on the closed
    contour.

    """
    positions = util.asarray_of_rows(positions)
    successors = np.roll(positions, -1, axis=0)
    d = [np.linalg.norm(b - a) for a, b in zip(positions, successors)]
    return np.array([np.mean(pair) for pair in zip(d, d[-1:] + d)])


def _rotate_array(positions, normals, n1, n2):
    """Rotate secondary sources from n1 to n2."""
    R = util.rotation_matrix(n1, n2)
    positions = np.inner(positions, R)
    normals = np.inner(normals, R)
    return positions, normals


def _linear_helper(ycoordinates, center, n0):
    """Create a full linear array from an array of y-coordinates."""
    N = len(ycoordinates)
    positions = np.zeros((N, 3))
    positions[:, 1] = ycoordinates - np.mean(ycoordinates[[0, -1]])
    positions, normals = _rotate_array(positions, [1, 0, 0], [1, 0, 0], n0)
    positions += center
    normals = np.tile(normals, (N, 1))
    weights = weights_linear(positions)
    return ArrayData(positions, normals, weights)


def concatenate(*arrays):
    """Concatenate ArrayData objects."""
    return ArrayData._make(np.concatenate(i) for i in zip(*arrays))
