"""Compute WFS driving functions.

.. include:: math-definitions.rst

.. plot::
    :context: reset

    import matplotlib.pyplot as plt
    import numpy as np
    import sfs
    from scipy.signal import unit_impulse

    # Plane wave
    npw = sfs.util.direction_vector(np.radians(-45))

    # Point source
    xs = -1.5, 1.5, 0
    rs = np.linalg.norm(xs)  # distance from origin
    ts = rs / sfs.default.c  # time-of-arrival at origin

    # Focused source
    xf = -0.5, 0.5, 0
    nf = sfs.util.direction_vector(np.radians(-45))  # normal vector
    rf = np.linalg.norm(xf)  # distance from origin
    tf = rf / sfs.default.c  # time-of-arrival at origin

    # Impulsive excitation
    fs = 44100
    signal = unit_impulse(512), fs

    # Circular loudspeaker array
    N = 32  # number of loudspeakers
    R = 1.5  # radius
    array = sfs.array.circular(N, R)

    grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.02)

    def plot(d, selection, secondary_source, t=0):
        p = sfs.td.synthesize(d, selection, array, secondary_source, grid=grid,
                              observation_time=t)
        sfs.plot2d.level(p, grid)
        sfs.plot2d.loudspeakers(array.x, array.n, selection * array.a, size=0.15)

"""
import numpy as _np
from numpy.core.umath_tests import inner1d as _inner1d

from . import apply_delays as _apply_delays
from . import secondary_source_point as _secondary_source_point
from .. import default as _default
from .. import util as _util


def plane_25d(x0, n0, n=[0, 1, 0], xref=[0, 0, 0], c=None):
    r"""Plane wave model by 2.5-dimensional WFS.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of secondary source orientations.
    n : (3,) array_like, optional
        Normal vector (propagation direction) of synthesized plane wave.
    xref : (3,) array_like, optional
        Reference position
    c : float, optional
        Speed of sound

    Returns
    -------
    delays : (N,) numpy.ndarray
        Delays of secondary sources in seconds.
    weights : (N,) numpy.ndarray
        Weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.td.synthesize()`.

    Notes
    -----
    2.5D correction factor

    .. math::

        g_0 = \sqrt{2 \pi |x_\mathrm{ref} - x_0|}

    d using a plane wave as source model

    .. math::

        d_{2.5D}(x_0,t) = h(t)
        2 g_0 \scalarprod{n}{n_0}
        \dirac{t - \frac{1}{c} \scalarprod{n}{x_0}}

    with wfs(2.5D) prefilter h(t), which is not implemented yet.

    See :sfs:`d_wfs/#equation-td-wfs-plane-25d`

    Examples
    --------
    .. plot::
        :context: close-figs

        delays, weights, selection, secondary_source = \
            sfs.td.wfs.plane_25d(array.x, array.n, npw)
        d = sfs.td.wfs.driving_signals(delays, weights, signal)
        plot(d, selection, secondary_source)

    """
    if c is None:
        c = _default.c
    x0 = _util.asarray_of_rows(x0)
    n0 = _util.asarray_of_rows(n0)
    n = _util.normalize_vector(n)
    xref = _util.asarray_1d(xref)
    g0 = _np.sqrt(2 * _np.pi * _np.linalg.norm(xref - x0, axis=1))
    delays = _inner1d(n, x0) / c
    weights = 2 * g0 * _inner1d(n, n0)
    selection = _util.source_selection_plane(n0, n)
    return delays, weights, selection, _secondary_source_point(c)


def point_25d(x0, n0, xs, xref=[0, 0, 0], c=None):
    r"""Point source by 2.5-dimensional WFS.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of secondary source orientations.
    xs : (3,) array_like
        Virtual source position.
    xref : (3,) array_like, optional
        Reference position
    c : float, optional
        Speed of sound

    Returns
    -------
    delays : (N,) numpy.ndarray
        Delays of secondary sources in seconds.
    weights: (N,) numpy.ndarray
        Weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.td.synthesize()`.

    Notes
    -----
    2.5D correction factor

    .. math::

         g_0 = \sqrt{2 \pi |x_\mathrm{ref} - x_0|}


    d using a point source as source model

    .. math::

         d_{2.5D}(x_0,t) = h(t)
         \frac{g_0  \scalarprod{(x_0 - x_s)}{n_0}}
         {2\pi |x_0 - x_s|^{3/2}}
         \dirac{t - \frac{|x_0 - x_s|}{c}}

    with wfs(2.5D) prefilter h(t), which is not implemented yet.

    See :sfs:`d_wfs/#equation-td-wfs-point-25d`

    Examples
    --------
    .. plot::
        :context: close-figs

        delays, weights, selection, secondary_source = \
            sfs.td.wfs.point_25d(array.x, array.n, xs)
        d = sfs.td.wfs.driving_signals(delays, weights, signal)
        plot(d, selection, secondary_source, t=ts)

    """
    if c is None:
        c = _default.c
    x0 = _util.asarray_of_rows(x0)
    n0 = _util.asarray_of_rows(n0)
    xs = _util.asarray_1d(xs)
    xref = _util.asarray_1d(xref)
    g0 = _np.sqrt(2 * _np.pi * _np.linalg.norm(xref - x0, axis=1))
    ds = x0 - xs
    r = _np.linalg.norm(ds, axis=1)
    delays = r/c
    weights = g0 * _inner1d(ds, n0) / (2 * _np.pi * r**(3/2))
    selection = _util.source_selection_point(n0, x0, xs)
    return delays, weights, selection, _secondary_source_point(c)


def focused_25d(x0, n0, xs, ns, xref=[0, 0, 0], c=None):
    r"""Point source by 2.5-dimensional WFS.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of secondary source orientations.
    xs : (3,) array_like
        Virtual source position.
    ns : (3,) array_like
        Normal vector (propagation direction) of focused source.
        This is used for secondary source selection,
        see `sfs.util.source_selection_focused()`.
    xref : (3,) array_like, optional
        Reference position
    c : float, optional
        Speed of sound

    Returns
    -------
    delays : (N,) numpy.ndarray
        Delays of secondary sources in seconds.
    weights: (N,) numpy.ndarray
        Weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.td.synthesize()`.

    Notes
    -----
    2.5D correction factor

    .. math::

         g_0 = \sqrt{\frac{|x_\mathrm{ref} - x_0|}
         {|x_0-x_s| + |x_\mathrm{ref}-x_0|}}


    d using a point source as source model

    .. math::

         d_{2.5D}(x_0,t) = h(t)
         \frac{g_0  \scalarprod{(x_0 - x_s)}{n_0}}
         {|x_0 - x_s|^{3/2}}
         \dirac{t + \frac{|x_0 - x_s|}{c}}

    with wfs(2.5D) prefilter h(t), which is not implemented yet.

    See :sfs:`d_wfs/#equation-td-wfs-focused-25d`

    Examples
    --------
    .. plot::
        :context: close-figs

        delays, weights, selection, secondary_source = \
            sfs.td.wfs.focused_25d(array.x, array.n, xf, nf)
        d = sfs.td.wfs.driving_signals(delays, weights, signal)
        plot(d, selection, secondary_source, t=tf)

    """
    if c is None:
        c = _default.c
    x0 = _util.asarray_of_rows(x0)
    n0 = _util.asarray_of_rows(n0)
    xs = _util.asarray_1d(xs)
    xref = _util.asarray_1d(xref)
    ds = x0 - xs
    r = _np.linalg.norm(ds, axis=1)
    g0 = _np.sqrt(_np.linalg.norm(xref - x0, axis=1)
                  / (_np.linalg.norm(xref - x0, axis=1) + r))
    delays = -r/c
    weights = g0 * _inner1d(ds, n0) / (2 * _np.pi * r**(3/2))
    selection = _util.source_selection_focused(ns, x0, xs)
    return delays, weights, selection, _secondary_source_point(c)


def driving_signals(delays, weights, signal):
    """Get driving signals per secondary source.

    Returned signals are the delayed and weighted mono input signal
    (with N samples) per channel (C).

    Parameters
    ----------
    delays : (C,) array_like
        Delay in seconds for each channel, negative values allowed.
    weights : (C,) array_like
        Amplitude weighting factor for each channel.
    signal : (N,) array_like + float
        Excitation signal consisting of (mono) audio data and a sampling
        rate (in Hertz).  A `DelayedSignal` object can also be used.

    Returns
    -------
    `DelayedSignal`
        A tuple containing the driving signals (in a `numpy.ndarray`
        with shape ``(N, C)``), followed by the sampling rate (in Hertz)
        and a (possibly negative) time offset (in seconds).

    """
    delays = _util.asarray_1d(delays)
    weights = _util.asarray_1d(weights)
    data, samplerate, signal_offset = _apply_delays(signal, delays)
    return _util.DelayedSignal(data * weights, samplerate, signal_offset)
