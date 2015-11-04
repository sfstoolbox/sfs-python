"""Compute positions of various secondary source distributions.

.. plot::
    :context: reset

    import sfs
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = 8, 4.5  # inch
    plt.rcParams['axes.grid'] = True

.. autoclass:: ArrayData
   :members: take

"""
from __future__ import division  # for Python 2.x
from collections import namedtuple
import numpy as np
from . import util


class ArrayData(namedtuple('ArrayData', 'x n a')):
    """Named tuple returned by array functions.

    See :obj:`collections.namedtuple`.

    Attributes
    ----------
    x : (N, 3) numpy.ndarray
        Positions of secondary sources
    n : (N, 3) numpy.ndarray
        Orientations (normal vectors) of secondary sources
    a : (N,) numpy.ndarray
        Weights of secondary sources

    """

    __slots__ = ()

    def __repr__(self):
        return 'ArrayData(\n' + ',\n'.join(
            '    {0}={1}'.format(name, repr(data).replace('\n', '\n      '))
            for name, data in zip('xna', self)) + ')'

    def take(self, indices):
        """Return a sub-array given by `indices`."""
        return ArrayData(self.x[indices], self.n[indices], self.a[indices])


def linear(N, spacing, center=[0, 0, 0], orientation=[1, 0, 0]):
    """Linear secondary source distribution.

    Parameters
    ----------
    N : int
        Number of secondary sources.
    spacing : float
        Distance (in metres) between secondary sources.
    center : (3,) array_like, optional
        Coordinates of array center.
    orientation : (3,) array_like, optional
        Orientation of the array.  By default, the loudspeakers have
        their main axis pointing into positive x-direction.

    Returns
    -------
    ArrayData
        Positions, orientations and weights of secondary sources.
        See :class:`ArrayData`.

    Example
    -------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.linear(16, 0.2, orientation=[0, -1, 0])
        sfs.plot.loudspeaker_2d(x0, n0, a0)
        plt.axis('equal')

    """
    return _linear_helper(np.arange(N) * spacing, center, orientation)


def linear_diff(distances, center=[0, 0, 0], orientation=[1, 0, 0]):
    """Linear secondary source distribution from a list of distances.

    Parameters
    ----------
    distances : (N-1,) array_like
        Sequence of secondary sources distances in metres.
    center, orientation
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

        x0, n0, a0 = sfs.array.linear_diff(4 * [0.3] + 6 * [0.15] + 4 * [0.3],
                                           orientation=[0, -1, 0])
        sfs.plot.loudspeaker_2d(x0, n0, a0)
        plt.axis('equal')

    """
    distances = util.asarray_1d(distances)
    ycoordinates = np.concatenate(([0], np.cumsum(distances)))
    return _linear_helper(ycoordinates, center, orientation)


def linear_random(N, min_spacing, max_spacing, center=[0, 0, 0],
                  orientation=[1, 0, 0], seed=None):
    """Randomly sampled linear array.

    Parameters
    ----------
    N : int
        Number of secondary sources.
    min_spacing, max_spacing : float
        Minimal and maximal distance (in metres) between secondary
        sources.
    center, orientation
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

        x0, n0, a0 = sfs.array.linear_random(12, 0.15, 0.4, orientation=[0, -1, 0])
        sfs.plot.loudspeaker_2d(x0, n0, a0)
        plt.axis('equal')

    """
    r = np.random.RandomState(seed)
    distances = r.uniform(min_spacing, max_spacing, size=N-1)
    return linear_diff(distances, center, orientation)


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


def rectangular(N, spacing, center=[0, 0, 0], orientation=[1, 0, 0]):
    """Rectangular secondary source distribution.

    Parameters
    ----------
    N : int or pair of int
        Number of secondary sources on each side of the rectangle.
        If a pair of numbers is given, the first one specifies the first
        and third segment, the second number specifies the second and
        fourth segment.
    spacing : float
        Distance (in metres) between secondary sources.
    center, orientation
        See :func:`linear`.  The `orientation` corresponds to the first
        linear segment.

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
    offset1 = spacing * (N2 - 1) / 2 + spacing / np.sqrt(2)
    offset2 = spacing * (N1 - 1) / 2 + spacing / np.sqrt(2)
    positions, normals, weights = concatenate(
        linear(N1, spacing, [-offset1, 0, 0], [1, 0, 0]),  # left
        linear(N2, spacing, [0, offset2, 0], [0, -1, 0]),  # upper
        linear(N1, spacing, [offset1, 0, 0], [-1, 0, 0]),  # right
        linear(N2, spacing, [0, -offset2, 0], [0, 1, 0]),  # lower
    )
    positions, normals = _rotate_array(positions, normals,
                                       [1, 0, 0], orientation)
    positions += center
    return ArrayData(positions, normals, weights)


def rounded_edge(Nxy, Nr, dx, center=[0, 0, 0], orientation=[1, 0, 0]):
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
    orientation : (3,) array_like, optional
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
                           orientation=[0, 1, 0])
    x00 = np.flipud(x00)
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))

    # rotate array
    positions, directions = _rotate_array(positions, directions,
                                          [1, 0, 0], orientation)
    # shift array to desired position
    positions += center
    return ArrayData(positions, directions, weights)


def planar(N, spacing, center=[0, 0, 0], orientation=[1, 0, 0]):
    """Planar secondary source distribtion.

    Parameters
    ----------
    N : int or pair of int
        Number of secondary sources along each edge.
        If a pair of numbers is given, the first one specifies the
        number on the horizontal edge, the second one specifies the
        number on the vertical edge.
    spacing : float
        Distance (in metres) between secondary sources.
    center, orientation
        See :func:`linear`.

    Returns
    -------
    ArrayData
        Positions, orientations and weights of secondary sources.
        See :class:`ArrayData`.

    """
    N1, N2 = (N, N) if np.isscalar(N) else N
    zcoordinates = np.arange(N2) * spacing
    zcoordinates -= np.mean(zcoordinates[[0, -1]])  # move center to origin
    subarrays = [linear(N1, spacing, center=[0, 0, z]) for z in zcoordinates]
    positions, normals, weights = concatenate(*subarrays)
    weights *= spacing
    positions, normals = _rotate_array(positions, normals,
                                       [1, 0, 0], orientation)
    positions += center
    return ArrayData(positions, normals, weights)


def cube(N, spacing, center=[0, 0, 0], orientation=[1, 0, 0]):
    """Cube-shaped secondary source distribtion.

    Parameters
    ----------
    N : int or triple of int
        Number of secondary sources along each edge.  If a triple of
        numbers is given, the first two specify the edges like in
        :func:`rectangular`, the last one specifies the vertical edge.
    spacing : float
        Distance (in metres) between secondary sources.
    center, orientation
        See :func:`linear`.  The `orientation` corresponds to the first
        planar segment.

    Returns
    -------
    ArrayData
        Positions, orientations and weights of secondary sources.
        See :class:`ArrayData`.

    """
    N1, N2, N3 = (N, N, N) if np.isscalar(N) else N
    offset1 = spacing * (N2 - 1) / 2 + spacing / np.sqrt(2)
    offset2 = spacing * (N1 - 1) / 2 + spacing / np.sqrt(2)
    offset3 = spacing * (N3 - 1) / 2 + spacing / np.sqrt(2)
    positions, directions, weights = concatenate(
        planar((N1, N3), spacing, [-offset1, 0, 0], [1, 0, 0]),  # west
        planar((N2, N3), spacing, [0, offset2, 0], [0, -1, 0]),  # north
        planar((N1, N3), spacing, [offset1, 0, 0], [-1, 0, 0]),  # east
        planar((N2, N3), spacing, [0, -offset2, 0], [0, 1, 0]),  # south
        planar((N2, N1), spacing, [0, 0, -offset3], [0, 0, 1]),  # bottom
        planar((N2, N1), spacing, [0, 0, offset3], [0, 0, -1]),  # top
    )
    positions, directions = _rotate_array(positions, directions,
                                          [1, 0, 0], orientation)
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


def load(fname, center=[0, 0, 0], orientation=[1, 0, 0]):
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
    positions, normals = _rotate_array(positions, normals,
                                       [1, 0, 0], orientation)
    positions += center
    return ArrayData(positions, normals, weights)


def weights_midpoint(positions, closed):
    """Calculate loudspeaker weights for a simply connected array.

    The weights are calculated according to the midpoint rule.


    Parameters
    ----------
    positions : (N, 3) array_like
        Sequence of secondary source positions.

        .. note:: The loudspeaker positions have to be ordered on the
                  contour!

    closed : bool
        ``True`` if the loudspeaker contour is closed.

    Returns
    -------
    (N,) numpy.ndarray
        Weights of secondary sources.

    """
    positions = util.asarray_of_rows(positions)
    if closed:
        before, after = -1, 0  # cyclic
    else:
        before, after = 1, -2  # mirrored
    positions = np.row_stack((positions[before], positions, positions[after]))
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return (distances[:-1] + distances[1:]) / 2


def _rotate_array(positions, normals, n1, n2):
    """Rotate secondary sources from n1 to n2."""
    R = util.rotation_matrix(n1, n2)
    positions = np.inner(positions, R)
    normals = np.inner(normals, R)
    return positions, normals


def _linear_helper(ycoordinates, center, orientation):
    """Create a full linear array from an array of y-coordinates."""
    N = len(ycoordinates)
    positions = np.zeros((N, 3))
    positions[:, 1] = ycoordinates - np.mean(ycoordinates[[0, -1]])
    positions, normals = _rotate_array(positions, [1, 0, 0],
                                       [1, 0, 0], orientation)
    positions += center
    normals = np.tile(normals, (N, 1))
    weights = weights_midpoint(positions, closed=False)
    return ArrayData(positions, normals, weights)


def concatenate(*arrays):
    """Concatenate :class:`ArrayData` objects."""
    return ArrayData._make(np.concatenate(i) for i in zip(*arrays))
