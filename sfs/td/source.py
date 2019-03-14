"""Compute the sound field generated by a sound source.

The Green's function describes the spatial sound propagation over time.

.. include:: math-definitions.rst

.. plot::
    :context: reset

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import unit_impulse
    import sfs

    xs = 1.5, 1, 0  # source position
    rs = np.linalg.norm(xs)  # distance from origin
    ts = rs / sfs.default.c  # time-of-arrival at origin

    # Impulsive excitation
    fs = 44100
    signal = unit_impulse(512), fs

    grid = sfs.util.xyz_grid([-2, 3], [-1, 2], 0, spacing=0.02)

"""

import numpy as np
from .. import util
from .. import default


def point(xs, signal, observation_time, grid, c=None):
    r"""Source model for a point source: 3D Green's function.

    Calculates the scalar sound pressure field for a given point in
    time, evoked by source excitation signal.

    Parameters
    ----------
    xs : (3,) array_like
        Position of source in cartesian coordinates.
    signal : (N,) array_like + float
        Excitation signal consisting of (mono) audio data and a sampling
        rate (in Hertz).  A `DelayedSignal` object can also be used.
    observation_time : float
        Observed point in time.
    grid : triple of array_like
        The grid that is used for the sound field calculations.
        See `sfs.util.xyz_grid()`.
    c : float, optional
        Speed of sound.

    Returns
    -------
    numpy.ndarray
        Scalar sound pressure field, evaluated at positions given by
        *grid*.

    Notes
    -----
    .. math::

        g(x-x_s,t) = \frac{1}{4 \pi |x - x_s|} \dirac{t - \frac{|x -
        x_s|}{c}}

    Examples
    --------
    .. plot::
        :context: close-figs

        p = sfs.td.source.point(xs, signal, ts, grid)
        sfs.plot.level(p, grid)

    """
    xs = util.asarray_1d(xs)
    data, samplerate, signal_offset = util.as_delayed_signal(signal)
    data = util.asarray_1d(data)
    grid = util.as_xyz_components(grid)
    if c is None:
        c = default.c
    r = np.linalg.norm(grid - xs)
    # evaluate g over grid
    weights = 1 / (4 * np.pi * r)
    delays = r / c
    base_time = observation_time - signal_offset
    return weights * np.interp(base_time - delays,
                               np.arange(len(data)) / samplerate,
                               data, left=0, right=0)


def point_image_sources(x0, signal, observation_time, grid, L, max_order,
                        coeffs=None, c=None):
    """Point source in a rectangular room using the mirror image source model.

    Parameters
    ----------
    x0 : (3,) array_like
        Position of source in cartesian coordinates.
    signal : (N,) array_like + float
        Excitation signal consisting of (mono) audio data and a sampling
        rate (in Hertz).  A `DelayedSignal` object can also be used.
    observation_time : float
        Observed point in time.
    grid : triple of array_like
        The grid that is used for the sound field calculations.
        See `sfs.util.xyz_grid()`.
    L : (3,) array_like
        Dimensions of the rectangular room.
    max_order : int
        Maximum number of reflections for each image source.
    coeffs : (6,) array_like, optional
        Reflection coeffecients of the walls.
        If not given, the reflection coefficients are set to one.
    c : float, optional
        Speed of sound.

    Returns
    -------
    numpy.ndarray
        Scalar sound pressure field, evaluated at positions given by
        *grid*.

    Examples
    --------
    .. plot::
        :context: close-figs

        room = 5, 3, 1.5  # room dimensions
        order = 2  # image source order
        coeffs = .8, .8, .6, .6, .7, .7  # wall reflection coefficients
        grid = sfs.util.xyz_grid([0, room[0]], [0, room[1]], 0, spacing=0.01)
        p = sfs.td.source.point_image_sources(
                xs, signal, 1.5 * ts, grid, room, order, coeffs)
        sfs.plot.level(p, grid)

    """
    if coeffs is None:
        coeffs = np.ones(6)

    positions, order = util.image_sources_for_box(x0, L, max_order)
    source_strengths = np.prod(coeffs**order, axis=1)

    p = 0
    for position, strength in zip(positions, source_strengths):
        if strength != 0:
            p += strength * point(position, signal, observation_time, grid, c)

    return p