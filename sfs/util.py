"""Various utility functions."""

import numpy as np
from . import defs


def rotation_matrix(n1, n2):
    """Compute rotation matrix for rotation from `n1` to `n2`."""
    n1 = asarray_1d(n1)
    n2 = asarray_1d(n2)
    # no rotation required
    if all(n1 == n2):
        return np.eye(3)

    v = np.cross(n1, n2)
    s = np.linalg.norm(v)

    # check for rotation of 180deg around one axis
    if s == 0:
        rot = np.identity(3)
        for i in np.arange(3):
            if np.abs(n1[i]) > 0 and np.abs(n1[i]) > 0 and n1[i] == -n2[i]:
                rot[i, i] = -1
        return rot

    c = np.inner(n1, n2)
    vx = [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]

    return np.identity(3) + vx + np.dot(vx, vx) * (1 - c) / s ** 2


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


def asarray_of_arrays(a, **kwargs):
    """Convert the input to an array consisting of arrays.

    A one-dimensional :class:`numpy.ndarray` with `dtype=object` is
    returned, containing the elements of `a` as :class:`numpy.ndarray`
    (whose `dtype` and other options can be specified with `**kwargs`).

    """
    result = np.empty(len(a), dtype=object)
    for i, element in enumerate(a):
        result[i] = np.asarray(element, **kwargs)
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
        :func:`numpy.arange`.  See :func:`strict_arange`.
    **kwargs
        All further arguments are forwarded to :func:`strict_arange`.

    Returns
    -------
    list of numpy.ndarrays
        A grid that can be used for sound field calculations.

    See Also
    --------
    strict_arange, numpy.meshgrid

    """
    if np.isscalar(spacing):
        spacing = [spacing] * 3
    args = []
    for i, coord in enumerate([x, y, z]):
        if np.isscalar(coord):
            args.append(coord)
        else:
            start, stop = coord
            args.append(strict_arange(start, stop, spacing[i],
                                      endpoint=endpoint, **kwargs))
    return np.meshgrid(*args, sparse=True, copy=False)


def normalize(p, grid, xnorm):
    """Normalize sound field wrt position `xnorm`."""
    return p / level(p, grid, xnorm)


def level(p, grid, x):
    """Determine level at position `x` in the sound field `p`."""
    x = asarray_1d(x)
    r = np.linalg.norm(grid - x)
    idx = np.unravel_index(r.argmin(), r.shape)
    # p is normally squeezed, therefore we need only 2 dimensions:
    idx = idx[:p.ndim]
    return abs(p[idx])


def broadcast_zip(*args):
    """Broadcast arguments to the same shape and then use :func:`zip`."""
    return zip(*np.broadcast_arrays(*args))
