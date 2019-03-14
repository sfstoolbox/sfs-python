"""Compute positions of various secondary source distributions.

.. plot::
    :context: reset

    import sfs
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = 8, 4.5  # inch
    plt.rcParams['axes.grid'] = True

"""
from collections import namedtuple
import numpy as np
from . import util


class SecondarySourceDistribution(namedtuple('SecondarySourceDistribution',
                                             'x n a')):
    """Named tuple returned by array functions.

    See `collections.namedtuple`.

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
        return 'SecondarySourceDistribution(\n' + ',\n'.join(
            '    {}={}'.format(name, repr(data).replace('\n', '\n      '))
            for name, data in zip('xna', self)) + ')'

    def take(self, indices):
        """Return a sub-array given by *indices*."""
        return SecondarySourceDistribution(
            self.x[indices], self.n[indices], self.a[indices])


def as_secondary_source_distribution(arg, **kwargs):
    r"""Create a `SecondarySourceDistribution`.

    Parameters
    ----------
    arg : sequence of between 1 and 3 array_like objects
        All elements are converted to NumPy arrays.
        If only 1 element is given, all normal vectors are set to *NaN*.
        If only 1 or 2 elements are given, all weights are set to ``1.0``.
    **kwargs
        All keyword arguments are forwarded to :func:`numpy.asarray`.

    Returns
    -------
    `SecondarySourceDistribution`
        A named tuple consisting of three `numpy.ndarray`\s containing
        positions, normal vectors and weights.

    """
    if len(arg) == 3:
        x, n, a = arg
    elif len(arg) == 2:
        x, n = arg
        a = 1.0
    elif len(arg) == 1:
        x, = arg
        n = np.nan, np.nan, np.nan
        a = 1.0
    else:
        raise TypeError('Between 1 and 3 elements are required')
    x = util.asarray_of_rows(x, **kwargs)
    n = util.asarray_of_rows(n, **kwargs)
    if len(n) == 1:
        n = np.tile(n, (len(x), 1))
    a = util.asarray_1d(a, **kwargs)
    if len(a) == 1:
        a = np.tile(a, len(x))
    return SecondarySourceDistribution(x, n, a)


def linear(N, spacing, *, center=[0, 0, 0], orientation=[1, 0, 0]):
    """Return linear, equidistantly sampled secondary source distribution.

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
    `SecondarySourceDistribution`
        Positions, orientations and weights of secondary sources.

    Examples
    --------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.linear(16, 0.2, orientation=[0, -1, 0])
        sfs.plot2d.loudspeakers(x0, n0, a0)
        plt.axis('equal')
        plt.xlabel('x / m')
        plt.ylabel('y / m')

    """
    return _linear_helper(np.arange(N) * spacing, center, orientation)


def linear_diff(distances, *, center=[0, 0, 0], orientation=[1, 0, 0]):
    """Return linear secondary source distribution from a list of distances.

    Parameters
    ----------
    distances : (N-1,) array_like
        Sequence of secondary sources distances in metres.
    center, orientation
        See `linear()`.

    Returns
    -------
    `SecondarySourceDistribution`
        Positions, orientations and weights of secondary sources.

    Examples
    --------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.linear_diff(4 * [0.3] + 6 * [0.15] + 4 * [0.3],
                                           orientation=[0, -1, 0])
        sfs.plot2d.loudspeakers(x0, n0, a0)
        plt.axis('equal')
        plt.xlabel('x / m')
        plt.ylabel('y / m')

    """
    distances = util.asarray_1d(distances)
    ycoordinates = np.concatenate(([0], np.cumsum(distances)))
    return _linear_helper(ycoordinates, center, orientation)


def linear_random(N, min_spacing, max_spacing, *, center=[0, 0, 0],
                  orientation=[1, 0, 0], seed=None):
    """Return randomly sampled linear array.

    Parameters
    ----------
    N : int
        Number of secondary sources.
    min_spacing, max_spacing : float
        Minimal and maximal distance (in metres) between secondary
        sources.
    center, orientation
        See `linear()`.
    seed : {None, int, array_like}, optional
        Random seed.  See `numpy.random.RandomState`.

    Returns
    -------
    `SecondarySourceDistribution`
        Positions, orientations and weights of secondary sources.

    Examples
    --------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.linear_random(
            N=12,
            min_spacing=0.15, max_spacing=0.4,
            orientation=[0, -1, 0])
        sfs.plot2d.loudspeakers(x0, n0, a0)
        plt.axis('equal')
        plt.xlabel('x / m')
        plt.ylabel('y / m')

    """
    r = np.random.RandomState(seed)
    distances = r.uniform(min_spacing, max_spacing, size=N-1)
    return linear_diff(distances, center=center, orientation=orientation)


def circular(N, R, *, center=[0, 0, 0]):
    """Return circular secondary source distribution parallel to the xy-plane.

    Parameters
    ----------
    N : int
        Number of secondary sources.
    R : float
        Radius in metres.
    center
        See `linear()`.

    Returns
    -------
    `SecondarySourceDistribution`
        Positions, orientations and weights of secondary sources.

    Examples
    --------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.circular(16, 1)
        sfs.plot2d.loudspeakers(x0, n0, a0, size=0.2, show_numbers=True)
        plt.axis('equal')
        plt.xlabel('x / m')
        plt.ylabel('y / m')

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
    return SecondarySourceDistribution(positions, normals, weights)


def rectangular(N, spacing, *, center=[0, 0, 0], orientation=[1, 0, 0]):
    """Return rectangular secondary source distribution.

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
        See `linear()`.  The *orientation* corresponds to the first
        linear segment.

    Returns
    -------
    `SecondarySourceDistribution`
        Positions, orientations and weights of secondary sources.

    Examples
    --------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.rectangular((4, 8), 0.2)
        sfs.plot2d.loudspeakers(x0, n0, a0, show_numbers=True)
        plt.axis('equal')
        plt.xlabel('x / m')
        plt.ylabel('y / m')

    """
    N1, N2 = (N, N) if np.isscalar(N) else N
    offset1 = spacing * (N2 - 1) / 2 + spacing / np.sqrt(2)
    offset2 = spacing * (N1 - 1) / 2 + spacing / np.sqrt(2)
    positions, normals, weights = concatenate(
        # left
        linear(N1, spacing, center=[-offset1, 0, 0], orientation=[1, 0, 0]),
        # upper
        linear(N2, spacing, center=[0, offset2, 0], orientation=[0, -1, 0]),
        # right
        linear(N1, spacing, center=[offset1, 0, 0], orientation=[-1, 0, 0]),
        # lower
        linear(N2, spacing, center=[0, -offset2, 0], orientation=[0, 1, 0]),
    )
    positions, normals = _rotate_array(positions, normals,
                                       [1, 0, 0], orientation)
    positions += center
    return SecondarySourceDistribution(positions, normals, weights)


def rounded_edge(Nxy, Nr, dx, *, center=[0, 0, 0], orientation=[1, 0, 0]):
    """Return SSD along the xy-axis with rounded edge at the origin.

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
    `SecondarySourceDistribution`
        Positions, orientations and weights of secondary sources.

    Examples
    --------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.rounded_edge(8, 5, 0.2)
        sfs.plot2d.loudspeakers(x0, n0, a0)
        plt.axis('equal')
        plt.xlabel('x / m')
        plt.ylabel('y / m')

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
    return SecondarySourceDistribution(positions, directions, weights)


def edge(Nxy, dx, *, center=[0, 0, 0], orientation=[1, 0, 0]):
    """Return SSD along the xy-axis with sharp edge at the origin.

    Parameters
    ----------
    Nxy : int
        Number of secondary sources along x- and y-axis.
    center : (3,) array_like, optional
        Position of edge.
    orientation : (3,) array_like, optional
        Normal vector of array.  Default orientation is along xy-axis.

    Returns
    -------
    `SecondarySourceDistribution`
        Positions, orientations and weights of secondary sources.

    Examples
    --------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.edge(8, 0.2)
        sfs.plot2d.loudspeakers(x0, n0, a0)
        plt.axis('equal')
        plt.xlabel('x / m')
        plt.ylabel('y / m')

    """
    # array along y-axis
    x00, n00, a00 = linear(Nxy, dx, center=[0, Nxy//2*dx+dx/2, 0])
    x00 = np.flipud(x00)
    positions = x00
    directions = n00
    weights = a00

    # array along x-axis
    x00, n00, a00 = linear(Nxy, dx, center=[Nxy//2*dx-dx/2, 0, 0],
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
    return SecondarySourceDistribution(positions, directions, weights)


def planar(N, spacing, *, center=[0, 0, 0], orientation=[1, 0, 0]):
    """Return planar secondary source distribtion.

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
        See `linear()`.

    Returns
    -------
    `SecondarySourceDistribution`
        Positions, orientations and weights of secondary sources.

    Examples
    --------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.planar(
            (4,3), 0.5, orientation=[0, 0, 1])  # 4 sources along y, 3 sources along x
        x0, n0, a0 = sfs.array.planar(
            (4,3), 0.5, orientation=[1, 0, 0])  # 4 sources along y, 3 sources along z

        x0, n0, a0 = sfs.array.planar(
            (4,3), 0.5, orientation=[0, 1, 0])  # 4 sources along x, 3 sources along z
        sfs.plot2d.loudspeakers(x0, n0, a0)  # plot the last ssd in 2D
        plt.axis('equal')
        plt.xlabel('x / m')
        plt.ylabel('y / m')


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
    return SecondarySourceDistribution(positions, normals, weights)


def cube(N, spacing, *, center=[0, 0, 0], orientation=[1, 0, 0]):
    """Return cube-shaped secondary source distribtion.

    Parameters
    ----------
    N : int or triple of int
        Number of secondary sources along each edge.  If a triple of
        numbers is given, the first two specify the edges like in
        `rectangular()`, the last one specifies the vertical edge.
    spacing : float
        Distance (in metres) between secondary sources.
    center, orientation
        See `linear()`.  The *orientation* corresponds to the first
        planar segment.

    Returns
    -------
    `SecondarySourceDistribution`
        Positions, orientations and weights of secondary sources.

    Examples
    --------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.cube(
            N=2, spacing=0.5,
            center=[0, 0, 0], orientation=[1, 0, 0])
        sfs.plot2d.loudspeakers(x0, n0, a0)
        plt.axis('equal')
        plt.xlabel('x / m')
        plt.ylabel('y / m')
        plt.title('view onto xy-plane')

    """
    N1, N2, N3 = (N, N, N) if np.isscalar(N) else N
    d = spacing
    offset1 = d * (N2 - 1) / 2 + d / np.sqrt(2)
    offset2 = d * (N1 - 1) / 2 + d / np.sqrt(2)
    offset3 = d * (N3 - 1) / 2 + d / np.sqrt(2)
    positions, directions, weights = concatenate(
        # west
        planar((N1, N3), d, center=[-offset1, 0, 0], orientation=[1, 0, 0]),
        # north
        planar((N2, N3), d, center=[0, offset2, 0], orientation=[0, -1, 0]),
        # east
        planar((N1, N3), d, center=[offset1, 0, 0], orientation=[-1, 0, 0]),
        # south
        planar((N2, N3), d, center=[0, -offset2, 0], orientation=[0, 1, 0]),
        # bottom
        planar((N2, N1), d, center=[0, 0, -offset3], orientation=[0, 0, 1]),
        # top
        planar((N2, N1), d, center=[0, 0, offset3], orientation=[0, 0, -1]),
    )
    positions, directions = _rotate_array(positions, directions,
                                          [1, 0, 0], orientation)
    positions += center
    return SecondarySourceDistribution(positions, directions, weights)


def sphere_load(file, radius, *, center=[0, 0, 0]):
    """Load spherical secondary source distribution from file.

    ASCII Format (see MATLAB SFS Toolbox) with 4 numbers (3 for the cartesian
    position vector, 1 for the integration weight) per secondary source located
    on the unit circle which is resized by the given radius and shifted to the
    given center.

    Returns
    -------
    `SecondarySourceDistribution`
        Positions, orientations and weights of secondary sources.

    Examples
    --------
    content of ``example_array_6LS_3D.txt``::

        1 0 0 1
        -1 0 0 1
        0 1 0 1
        0 -1 0 1
        0 0 1 1
        0 0 -1 1

    corresponds to the `3-dimensional 6-point spherical 3-design
    <http://neilsloane.com/sphdesigns/dim3/des.3.6.3.txt>`_.

    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.sphere_load(
            '../data/arrays/example_array_6LS_3D.txt',
            radius=2,
            center=[0, 0, 0])
        sfs.plot2d.loudspeakers(x0, n0, a0, size=0.25)
        plt.axis('equal')
        plt.xlabel('x / m')
        plt.ylabel('y / m')
        plt.title('view onto xy-plane')

    """
    data = np.loadtxt(file)
    positions, weights = data[:, :3], data[:, 3]
    normals = -positions
    positions *= radius
    positions += center
    return SecondarySourceDistribution(positions, normals, weights)


def load(file, *, center=[0, 0, 0], orientation=[1, 0, 0]):
    """Load secondary source distribution from file.

    Comma Separated Values (CSV) format with 7 values
    (3 for the cartesian position vector, 3 for the cartesian inward normal
    vector, 1 for the integration weight) per secondary source.

    Returns
    -------
    `SecondarySourceDistribution`
        Positions, orientations and weights of secondary sources.

    Examples
    --------
    content of ``example_array_4LS_2D.csv``::

        1,0,0,-1,0,0,1
        0,1,0,0,-1,0,1
        -1,0,0,1,0,0,1
        0,-1,0,0,1,0,1

    corresponds to 4 sources at 1, j, -1, -j in the complex plane. This setup
    is typically used for Quadraphonic audio reproduction.

    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.load('../data/arrays/example_array_4LS_2D.csv')
        sfs.plot2d.loudspeakers(x0, n0, a0)
        plt.axis('equal')
        plt.xlabel('x / m')
        plt.ylabel('y / m')

    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.load(
            '../data/arrays/wfs_university_rostock_2018.csv')
        sfs.plot2d.loudspeakers(x0, n0, a0)
        plt.axis('equal')
        plt.xlabel('x / m')
        plt.ylabel('y / m')
        plt.title('top view of 64 channel WFS system at university of Rostock')

    """
    data = np.loadtxt(file, delimiter=',')
    positions, normals, weights = data[:, :3], data[:, 3:6], data[:, 6]
    positions, normals = _rotate_array(positions, normals,
                                       [1, 0, 0], orientation)
    positions += center
    return SecondarySourceDistribution(positions, normals, weights)


def weights_midpoint(positions, *, closed):
    """Calculate loudspeaker weights for a simply connected array.

    The weights are calculated according to the midpoint rule.


    Parameters
    ----------
    positions : (N, 3) array_like
        Sequence of secondary source positions.

        .. note:: The loudspeaker positions have to be ordered along the
                  contour.

    closed : bool
        ``True`` if the loudspeaker contour is closed.

    Returns
    -------
    (N,) numpy.ndarray
        Weights of secondary sources.

    Examples
    --------
    >>> import sfs
    >>> x0, n0, a0 = sfs.array.circular(2**5, 1)
    >>> a = sfs.array.weights_midpoint(x0, closed=True)
    >>> max(abs(a0-a))
    0.0003152601902411123

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
    center = util.asarray_1d(center)
    N = len(ycoordinates)
    positions = np.zeros((N, 3))
    positions[:, 1] = ycoordinates - np.mean(ycoordinates[[0, -1]])
    positions, normals = _rotate_array(positions, [1, 0, 0],
                                       [1, 0, 0], orientation)
    positions += center
    normals = np.tile(normals, (N, 1))
    weights = weights_midpoint(positions, closed=False)
    return SecondarySourceDistribution(positions, normals, weights)


def concatenate(*arrays):
    """Concatenate `SecondarySourceDistribution` objects.

    Returns
    -------
    `SecondarySourceDistribution`
        Positions, orientations and weights
        of the concatenated secondary sources.

    Examples
    --------
    .. plot::
        :context: close-figs

        ssd1 = sfs.array.edge(10, 0.2)
        ssd2 = sfs.array.edge(20, 0.1, center=[2, 2, 0], orientation=[-1, 0, 0])
        x0, n0, a0 = sfs.array.concatenate(ssd1, ssd2)
        sfs.plot2d.loudspeakers(x0, n0, a0)
        plt.axis('equal')
        plt.xlabel('x / m')
        plt.ylabel('y / m')

    """
    return SecondarySourceDistribution._make(np.concatenate(i)
                                             for i in zip(*arrays))
