"""Various utility functions.

.. include:: math-definitions.rst

"""

import collections
import numpy as np
from numpy.core.umath_tests import inner1d
from scipy.special import spherical_jn, spherical_yn
from . import default


def rotation_matrix(n1, n2):
    """Compute rotation matrix for rotation from *n1* to *n2*.

    Parameters
    ----------
    n1, n2 : (3,) array_like
        Two vectors.  They don't have to be normalized.

    Returns
    -------
    (3, 3) `numpy.ndarray`
        Rotation matrix.

    """
    n1 = normalize_vector(n1)
    n2 = normalize_vector(n2)
    I = np.identity(3)
    if np.all(n1 == n2):
        return I  # no rotation
    elif np.all(n1 == -n2):
        return -I  # flip
    # TODO: check for *very close to* parallel vectors

    # Algorithm from http://math.stackexchange.com/a/476311
    v = v0, v1, v2 = np.cross(n1, n2)
    s = np.linalg.norm(v)  # sine
    c = np.inner(n1, n2)  # cosine
    vx = [[0, -v2, v1],
          [v2, 0, -v0],
          [-v1, v0, 0]]  # skew-symmetric cross-product matrix
    return I + vx + np.dot(vx, vx) * (1 - c) / s**2


def wavenumber(omega, c=None):
    """Compute the wavenumber for a given radial frequency."""
    if c is None:
        c = default.c
    return omega / c


def direction_vector(alpha, beta=np.pi/2):
    """Compute normal vector from azimuth, colatitude."""
    return sph2cart(alpha, beta, 1)


def sph2cart(alpha, beta, r):
    r"""Spherical to cartesian coordinate transform.

    .. math::

        x = r \cos \alpha \sin \beta \\
        y = r \sin \alpha \sin \beta \\
        z = r \cos \beta

    with :math:`\alpha \in [0, 2\pi), \beta \in [0, \pi], r \geq 0`

    Parameters
    ----------
    alpha : float or array_like
            Azimuth angle in radiants
    beta : float or array_like
            Colatitude angle in radiants (with 0 denoting North pole)
    r : float or array_like
            Radius

    Returns
    -------
    x : float or `numpy.ndarray`
        x-component of Cartesian coordinates
    y : float or `numpy.ndarray`
        y-component of Cartesian coordinates
    z : float or `numpy.ndarray`
        z-component of Cartesian coordinates

    """
    x = r * np.cos(alpha) * np.sin(beta)
    y = r * np.sin(alpha) * np.sin(beta)
    z = r * np.cos(beta)
    return x, y, z


def cart2sph(x, y, z):
    r"""Cartesian to spherical coordinate transform.

    .. math::

        \alpha = \arctan \left( \frac{y}{x} \right) \\
        \beta = \arccos \left( \frac{z}{r} \right) \\
        r = \sqrt{x^2 + y^2 + z^2}

    with :math:`\alpha \in [-pi, pi], \beta \in [0, \pi], r \geq 0`

    Parameters
    ----------
    x : float or array_like
        x-component of Cartesian coordinates
    y : float or array_like
        y-component of Cartesian coordinates
    z : float or array_like
        z-component of Cartesian coordinates

    Returns
    -------
    alpha : float or `numpy.ndarray`
            Azimuth angle in radiants
    beta : float or `numpy.ndarray`
            Colatitude angle in radiants (with 0 denoting North pole)
    r : float or `numpy.ndarray`
            Radius

    """
    r = np.sqrt(x**2 + y**2 + z**2)
    alpha = np.arctan2(y, x)
    beta = np.arccos(z / r)
    return alpha, beta, r


def asarray_1d(a, **kwargs):
    """Squeeze the input and check if the result is one-dimensional.

    Returns *a* converted to a `numpy.ndarray` and stripped of
    all singleton dimensions.  Scalars are "upgraded" to 1D arrays.
    The result must have exactly one dimension.
    If not, an error is raised.

    """
    result = np.squeeze(np.asarray(a, **kwargs))
    if result.ndim == 0:
        result = result.reshape((1,))
    elif result.ndim > 1:
        raise ValueError("array must be one-dimensional")
    return result


def asarray_of_rows(a, **kwargs):
    """Convert to 2D array, turn column vector into row vector.

    Returns *a* converted to a `numpy.ndarray` and stripped of
    all singleton dimensions.  If the result has exactly one dimension,
    it is re-shaped into a 2D row vector.

    """
    result = np.squeeze(np.asarray(a, **kwargs))
    if result.ndim == 1:
        result = result.reshape(1, -1)
    return result


def as_xyz_components(components, **kwargs):
    r"""Convert *components* to `XyzComponents` of `numpy.ndarray`\s.

    The *components* are first converted to NumPy arrays (using
    :func:`numpy.asarray`) which are then assembled into an
    `XyzComponents` object.

    Parameters
    ----------
    components : triple or pair of array_like
        The values to be used as X, Y and Z arrays.  Z is optional.
    **kwargs
        All further arguments are forwarded to :func:`numpy.asarray`,
        which is applied to the elements of *components*.

    """
    return XyzComponents([np.asarray(c, **kwargs) for c in components])


def as_delayed_signal(arg, **kwargs):
    """Make sure that the given argument can be used as a signal.

    Parameters
    ----------
    arg : sequence of 1 array_like followed by 1 or 2 scalars
        The first element is converted to a NumPy array, the second
        element is used as the sampling rate (in Hertz) and the optional
        third element is used as the starting time of the signal (in
        seconds).  Default starting time is 0.
    **kwargs
        All keyword arguments are forwarded to :func:`numpy.asarray`.

    Returns
    -------
    `DelayedSignal`
        A named tuple consisting of a `numpy.ndarray` containing the
        audio data, followed by the sampling rate (in Hertz) and the
        starting time (in seconds) of the signal.

    Examples
    --------
    Typically, this is used together with tuple unpacking to assign the
    audio data, the sampling rate and the starting time to separate
    variables:

    >>> import sfs
    >>> sig = [1], 44100
    >>> data, fs, signal_offset = sfs.util.as_delayed_signal(sig)
    >>> data
    array([1])
    >>> fs
    44100
    >>> signal_offset
    0

    """
    try:
        data, samplerate, *time = arg
        time, = time or [0]
    except (IndexError, TypeError, ValueError):
        pass
    else:
        valid_arguments = (not np.isscalar(data) and
                           np.isscalar(samplerate) and
                           np.isscalar(time))
        if valid_arguments:
            data = np.asarray(data, **kwargs)
            return DelayedSignal(data, samplerate, time)
    raise TypeError('expected audio data, samplerate, optional start time')


def strict_arange(start, stop, step=1, *, endpoint=False, dtype=None,
                  **kwargs):
    """Like :func:`numpy.arange`, but compensating numeric errors.

    Unlike :func:`numpy.arange`, but similar to :func:`numpy.linspace`,
    providing ``endpoint=True`` includes both endpoints.

    Parameters
    ----------
    start, stop, step, dtype
        See :func:`numpy.arange`.
    endpoint
        See :func:`numpy.linspace`.

        .. note:: With ``endpoint=True``, the difference between *start*
           and *end* value must be an integer multiple of the
           corresponding *spacing* value!
    **kwargs
        All further arguments are forwarded to :func:`numpy.isclose`.

    Returns
    -------
    `numpy.ndarray`
        Array of evenly spaced values.  See :func:`numpy.arange`.

    """
    remainder = (stop - start) % step
    if np.any(np.isclose(remainder, (0.0, step), **kwargs)):
        if endpoint:
            stop += step * 0.5
        else:
            stop -= step * 0.5
    elif endpoint:
        raise ValueError("Invalid stop value for endpoint=True")
    return np.arange(start, stop, step, dtype)


def xyz_grid(x, y, z, *, spacing, endpoint=True, **kwargs):
    """Create a grid with given range and spacing.

    Parameters
    ----------
    x, y, z : float or pair of float
        Inclusive range of the respective coordinate or a single value
        if only a slice along this dimension is needed.
    spacing : float or triple of float
        Grid spacing.  If a single value is specified, it is used for
        all dimensions, if multiple values are given, one value is used
        per dimension.  If a dimension (*x*, *y* or *z*) has only a
        single value, the corresponding spacing is ignored.
    endpoint : bool, optional
        If ``True`` (the default), the endpoint of each range is
        included in the grid.  Use ``False`` to get a result similar to
        :func:`numpy.arange`.  See `strict_arange()`.
    **kwargs
        All further arguments are forwarded to `strict_arange()`.

    Returns
    -------
    `XyzComponents`
        A grid that can be used for sound field calculations.

    See Also
    --------
    strict_arange, numpy.meshgrid

    """
    if np.isscalar(spacing):
        spacing = [spacing] * 3
    ranges = []
    scalars = []
    for i, coord in enumerate([x, y, z]):
        if np.isscalar(coord):
            scalars.append((i, coord))
        else:
            start, stop = coord
            ranges.append(strict_arange(start, stop, spacing[i],
                                        endpoint=endpoint, **kwargs))
    grid = np.meshgrid(*ranges, sparse=True, copy=False)
    for i, s in scalars:
        grid.insert(i, s)
    return XyzComponents(grid)


def normalize(p, grid, xnorm):
    """Normalize sound field wrt position *xnorm*."""
    return p / np.abs(probe(p, grid, xnorm))


def probe(p, grid, x):
    """Determine the value at position *x* in the sound field *p*."""
    grid = as_xyz_components(grid)
    x = asarray_1d(x)
    r = np.linalg.norm(grid - x)
    idx = np.unravel_index(r.argmin(), r.shape)
    return p[idx]


def broadcast_zip(*args):
    """Broadcast arguments to the same shape and then use :func:`zip`."""
    return zip(*np.broadcast_arrays(*args))


def normalize_vector(x):
    """Normalize a 1D vector."""
    x = asarray_1d(x)
    return x / np.linalg.norm(x)


def db(x, *, power=False):
    """Convert *x* to decibel.

    Parameters
    ----------
    x : array_like
        Input data.  Values of 0 lead to negative infinity.
    power : bool, optional
        If ``power=False`` (the default), *x* is squared before
        conversion.

    """
    with np.errstate(divide='ignore'):
        return (10 if power else 20) * np.log10(np.abs(x))


class XyzComponents(np.ndarray):
    """See __init__()."""

    def __init__(self, components):
        r"""Triple (or pair) of components: x, y, and optionally z.

        Instances of this class can be used to store coordinate grids
        (either regular grids like in `xyz_grid()` or arbitrary point
        clouds) or vector fields (e.g. particle velocity).

        This class is a subclass of `numpy.ndarray`.  It is
        one-dimensional (like a plain `list`) and has a length of 3 (or
        2, if no z-component is available).  It uses ``dtype=object`` in
        order to be able to store other `numpy.ndarray`\s of arbitrary
        shapes but also scalars, if needed.  Because it is a NumPy array
        subclass, it can be used in operations with scalars and "normal"
        NumPy arrays, as long as they have a compatible shape.  Like any
        NumPy array, instances of this class are iterable and can be
        used, e.g., in for-loops and tuple unpacking.  If slicing or
        broadcasting leads to an incompatible shape, a plain
        `numpy.ndarray` with ``dtype=object`` is returned.

        To make sure the *components* are NumPy arrays themselves, use
        `as_xyz_components()`.

        Parameters
        ----------
        components : (3,) or (2,) array_like
            The values to be used as X, Y and Z data.  Z is optional.

        """
        # This method does nothing, it's only here for the documentation!

    def __new__(cls, components):
        # object arrays cannot be created and populated in a single step:
        obj = np.ndarray.__new__(cls, len(components), dtype=object)
        for i, component in enumerate(components):
            obj[i] = component
        return obj

    def __array_finalize__(self, obj):
        if self.ndim == 0:
            pass  # this is allowed, e.g. for np.inner()
        elif self.ndim > 1 or len(self) not in (2, 3):
            raise ValueError("XyzComponents can only have 2 or 3 components")

    def __array_prepare__(self, obj, context=None):
        if obj.ndim == 1 and len(obj) in (2, 3):
            return obj.view(XyzComponents)
        return obj

    def __array_wrap__(self, obj, context=None):
        if obj.ndim != 1 or len(obj) not in (2, 3):
            return obj.view(np.ndarray)
        return obj

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if start == 0 and stop in (2, 3) and step == 1:
                return np.ndarray.__getitem__(self, index)
        # Slices other than xy and xyz are "downgraded" to ndarray
        return np.ndarray.__getitem__(self.view(np.ndarray), index)

    def __repr__(self):
        return 'XyzComponents(\n' + ',\n'.join(
            '    {}={}'.format(name, repr(data).replace('\n', '\n      '))
            for name, data in zip('xyz', self)) + ')'

    def make_property(index, doc):

        def getter(self):
            return self[index]

        def setter(self, value):
            self[index] = value

        return property(getter, setter, doc=doc)

    x = make_property(0, doc='x-component.')
    y = make_property(1, doc='y-component.')
    z = make_property(2, doc='z-component (optional).')

    del make_property

    def apply(self, func, *args, **kwargs):
        """Apply a function to each component.

        The function *func* will be called once for each component,
        passing the current component as first argument.  All further
        arguments are passed after that.
        The results are returned as a new `XyzComponents` object.

        """
        return XyzComponents([func(i, *args, **kwargs) for i in self])


DelayedSignal = collections.namedtuple('DelayedSignal', 'data samplerate time')
"""A tuple of audio data, sampling rate and start time.

This class (a `collections.namedtuple`) is not meant to be instantiated
by users.

To pass a signal to a function, just use a simple `tuple` or `list`
containing the audio data and the sampling rate (in Hertz), with an
optional starting time (in seconds) as a third item.
If you want to ensure that a given variable contains a valid signal, use
`sfs.util.as_delayed_signal()`.

"""


def image_sources_for_box(x, L, N, *, prune=True):
    """Image source method for a cuboid room.

    The classical method by Allen and Berkley :cite:`Allen1979`.

    Parameters
    ----------
    x : (D,) array_like
        Original source location within box.
        Values between 0 and corresponding side length.
    L : (D,) array_like
        side lengths of room.
    N : int
        Maximum number of reflections per image source, see below.
    prune : bool, optional
        selection of image sources:

        - If True (default):
          Returns all images reflected up to N times.
          This is the usual interpretation of N as "maximum order".

        - If False:
          Returns reflected up to N times between individual wall pairs,
          a total number of :math:`M := (2N+1)^D`.
          This larger set is useful e.g. to select image sources based on
          distance to listener, as suggested by :cite:`Borish1984`.


    Returns
    -------
    xs : (M, D) `numpy.ndarray`
        original & image source locations.
    wall_count : (M, 2D) `numpy.ndarray`
        number of reflections at individual walls for each source.

    """
    def _images_1d_unit_box(x, N):
        result = np.arange(-N, N + 1, dtype=x.dtype)
        result[N % 2::2] += x
        result[1 - (N % 2)::2] += 1 - x
        return result

    def _count_walls_1d(a):
        b = np.floor(a/2)
        c = np.ceil((a-1)/2)
        return np.abs(np.stack([b, c], axis=1)).astype(int)

    L = asarray_1d(L)
    x = asarray_1d(x)/L
    D = len(x)
    xs = [_images_1d_unit_box(coord, N) for coord in x]
    xs = np.reshape(np.transpose(np.meshgrid(*xs, indexing='ij')), (-1, D))

    wall_count = np.concatenate([_count_walls_1d(d) for d in xs.T], axis=1)
    xs *= L

    if prune is True:
        N_mask = np.sum(wall_count, axis=1) <= N
        xs = xs[N_mask, :]
        wall_count = wall_count[N_mask, :]

    return xs, wall_count


def spherical_hn2(n, z):
    r"""Spherical Hankel function of 2nd kind.

    Defined as https://dlmf.nist.gov/10.47.E6,

    .. math::

        \hankel{2}{n}{z} = \sqrt{\frac{\pi}{2z}}
        \Hankel{2}{n + \frac{1}{2}}{z},

    where :math:`\Hankel{2}{n}{\cdot}` is the Hankel function of the
    second kind and n-th order, and :math:`z` its complex argument.

    Parameters
    ----------
    n : array_like
        Order of the spherical Hankel function (n >= 0).
    z : array_like
        Argument of the spherical Hankel function.

    """
    return spherical_jn(n, z) - 1j * spherical_yn(n, z)


def source_selection_plane(n0, n):
    """Secondary source selection for a plane wave.

    Eq.(13) from :cite:`Spors2008`

    """
    n0 = asarray_of_rows(n0)
    n = normalize_vector(n)
    return np.inner(n, n0) >= default.selection_tolerance


def source_selection_point(n0, x0, xs):
    """Secondary source selection for a point source.

    Eq.(15) from :cite:`Spors2008`

    """
    n0 = asarray_of_rows(n0)
    x0 = asarray_of_rows(x0)
    xs = asarray_1d(xs)
    ds = x0 - xs
    return inner1d(ds, n0) >= default.selection_tolerance


def source_selection_line(n0, x0, xs):
    """Secondary source selection for a line source.

    compare Eq.(15) from :cite:`Spors2008`

    """
    return source_selection_point(n0, x0, xs)


def source_selection_focused(ns, x0, xs):
    """Secondary source selection for a focused source.

    Eq.(2.78) from :cite:`Wierstorf2014`

    """
    x0 = asarray_of_rows(x0)
    xs = asarray_1d(xs)
    ns = normalize_vector(ns)
    ds = xs - x0
    return inner1d(ns, ds) >= default.selection_tolerance


def source_selection_all(N):
    """Select all secondary sources."""
    return np.ones(N, dtype=bool)


def max_order_circular_harmonics(N):
    r"""Maximum order of 2D/2.5D HOA.

    It returns the maximum order for which no spatial aliasing appears.
    It is given on page 132 of [Ahrens2012]_ as

    .. math::
        \mathtt{max\_order} =
            \begin{cases}
                N/2 - 1 & \text{even}\;N \\
                (N-1)/2 & \text{odd}\;N,
            \end{cases}

    which is equivalent to

    .. math::
        \mathtt{max\_order} = \big\lfloor \frac{N - 1}{2} \big\rfloor.

    Parameters
    ----------
    N : int
        Number of secondary sources.

    """
    return (N - 1) // 2


def max_order_spherical_harmonics(N):
    r"""Maximum order of 3D HOA.

    .. math::
        \mathtt{max\_order} = \lfloor \sqrt{N} \rfloor - 1.

    Parameters
    ----------
    N : int
        Number of secondary sources.

    """
    return int(np.sqrt(N) - 1)
