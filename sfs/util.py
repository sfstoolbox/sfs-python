"""Various utility functions."""

import numpy as np
from . import defs


def rotation_matrix(n1, n2):
    """Compute rotation matrix for rotation from `n1` to `n2`.

    Parameters
    ----------
    n1, n2 : (3,) array_like
        Two vectors.  They don't have to be normalized.

    Returns
    -------
    (3, 3) numpy.ndarray
        Rotation matrix.

    """
    n1 = normal_vector(n1)
    n2 = normal_vector(n2)
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
    vx = np.matrix([[0, -v2, v1],
                    [v2, 0, -v0],
                    [-v1, v0, 0]])  # skew-symmetric cross-product matrix
    return I + vx + vx**2 * (1 - c) / s**2


def wavenumber(omega, c=None):
    """Compute the wavenumber for a given radial frequency."""
    if c is None:
        c = defs.c
    return omega / c


def direction_vector(alpha, beta=np.pi/2):
    """Compute normal vector from azimuth, colatitude."""
    return sph2cart(alpha, beta, 1)


def sph2cart(alpha, beta, r):
    """Spherical to cartesian coordinates."""
    x = r * np.cos(alpha) * np.sin(beta)
    y = r * np.sin(alpha) * np.sin(beta)
    z = r * np.cos(beta)

    return x, y, z


def cart2sph(x, y, z):
    """Cartesian to spherical coordinates."""
    alpha = np.arctan2(y, x)
    beta = np.arccos(z / np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)

    return alpha, beta, r


def asarray_1d(a, **kwargs):
    """Squeeze the input and check if the result is one-dimensional.

    Returns `a` converted to a :class:`numpy.ndarray` and stripped of
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

    Returns `a` converted to a :class:`numpy.ndarray` and stripped of
    all singleton dimensions.  If the result has exactly one dimension,
    it is re-shaped into a 2D row vector.

    """
    result = np.squeeze(np.asarray(a, **kwargs))
    if result.ndim == 1:
        result = result.reshape(1, -1)
    return result


def strict_arange(start, stop, step=1, endpoint=False, dtype=None, **kwargs):
    """Like :func:`numpy.arange`, but compensating numeric errors.

    Unlike :func:`numpy.arange`, but similar to :func:`numpy.linspace`,
    providing `endpoint=True` includes both endpoints.

    Parameters
    ----------
    start, stop, step, dtype
        See :func:`numpy.arange`.
    endpoint
        See :func:`numpy.linspace`.

        .. note:: With `endpoint=True`, the difference between `start`
           and `end` value must be an integer multiple of the
           corresponding `spacing` value!
    **kwargs
        All further arguments are forwarded to :func:`numpy.isclose`.

    Returns
    -------
    numpy.ndarray
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


def xyz_grid(x, y, z, spacing, endpoint=True, **kwargs):
    """Create a grid with given range and spacing.

    Parameters
    ----------
    x, y, z : float or pair of float
        Inclusive range of the respective coordinate or a single value
        if only a slice along this dimension is needed.
    spacing : float or triple of float
        Grid spacing.  If a single value is specified, it is used for
        all dimensions, if multiple values are given, one value is used
        per dimension.  If a dimension (`x`, `y` or `z`) has only a
        single value, the corresponding spacing is ignored.
    endpoint : bool, optional
        If ``True`` (the default), the endpoint of each range is
        included in the grid.  Use ``False`` to get a result similar to
        :func:`numpy.arange`.  See :func:`sfs.util.strict_arange`.
    **kwargs
        All further arguments are forwarded to
        :func:`sfs.util.strict_arange`.

    Returns
    -------
    XyzComponents
        A grid that can be used for sound field calculations.
        See :class:`sfs.util.XyzComponents`.

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
    """Normalize sound field wrt position `xnorm`."""
    return p / np.abs(probe(p, grid, xnorm))


def probe(p, grid, x):
    """Determine the value at position `x` in the sound field `p`."""
    grid = XyzComponents(grid)
    x = asarray_1d(x)
    r = np.linalg.norm(grid - x)
    idx = np.unravel_index(r.argmin(), r.shape)
    return p[idx]


def broadcast_zip(*args):
    """Broadcast arguments to the same shape and then use :func:`zip`."""
    return zip(*np.broadcast_arrays(*args))


def normal_vector(x):
    """Normalize a 1D vector."""
    x = asarray_1d(x)
    return x / np.linalg.norm(x)


def displacement(v, omega):
    """Particle displacement

    .. math::

        d(x, t) = \int_0^t v(x, t) dt

    """
    v = XyzComponents(v)
    return v / (1j * omega)


def db(x, power=False):
    """Convert `x` to decibel.

    Parameters
    ----------
    x : array_like
        Input data.  Values of 0 lead to negative infinity.
    power : bool, optional
        If `power=False` (the default), `x` is squared before
        conversion.

    """
    with np.errstate(divide='ignore'):
        return 10 if power else 20 * np.log10(np.abs(x))


class XyzComponents(np.ndarray):
    """See __init__()."""

    def __init__(self, components, **kwargs):
        """Triple (or pair) of arrays: x, y, and optionally z.

        Instances of this class can be used to store coordinate grids
        (either regular grids like in :func:`sfs.util.xyz_grid` or
        arbitrary point clouds) or vector fields (e.g. particle
        velocity).

        This class is a subclass of :class:`numpy.ndarray`.  It is
        one-dimensional (like a plain :class:`list`) and has a length of
        3 (or 2, if no z-component is available).  It uses
        `dtype=object` in order to be able to store other
        `numpy.ndarray`\s of arbitrary shapes but also scalars, if
        needed.  Because it is a NumPy array subclass, it can be used in
        operations with scalars and "normal" NumPy arrays, as long as
        they have a compatible shape.  Like any NumPy array, instances
        of this class are iterable and can be used, e.g., in for-loops
        and tuple unpacking.  If slicing or broadcasting leads to an
        incompatible shape, a plain `numpy.ndarray` with `dtype=object`
        is returned.

        Parameters
        ----------
        components : triple or pair of array_like
            The values to be used as X, Y and Z arrays.  Z is optional.
        **kwargs
            All further arguments are forwarded to
            :func:`numpy.asarray`, which is applied to the elements of
            `components`.

        """
        # This method does nothing, it's only here for the documentation!

    def __new__(cls, components, **kwargs):
        # object arrays cannot be created and populated in a single step:
        obj = np.ndarray.__new__(cls, len(components), dtype=object)
        for i, component in enumerate(components):
            obj[i] = np.asarray(component, **kwargs)
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
            '    {0}={1}'.format(name, repr(data).replace('\n', '\n      '))
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

        The function `func` will be called once for each component,
        passing the current component as first argument.  All further
        arguments are passed after that.
        The results are returned as a new :class:`XyzComponents` object.

        """
        return XyzComponents([func(i, *args, **kwargs) for i in self])
