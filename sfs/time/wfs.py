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
        p = sfs.time.synthesize(d, selection, array, secondary_source, grid=grid,
                                observation_time=t)
        sfs.plot.level(p, grid)
        sfs.plot.loudspeaker_2d(array.x, array.n, selection * array.a, size=0.15)

"""
import numpy as np
from numpy.core.umath_tests import inner1d  # element-wise inner product
from .. import default
from .. import util
from . import secondary_source_point, apply_delays


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
        single secondary source.  See `sfs.time.synthesize()`.

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

    References
    ----------
    See http://sfstoolbox.org/en/latest/#equation-d.wfs.pw.2.5D

    Examples
    --------
    .. plot::
        :context: close-figs

        delays, weights, selection, secondary_source = \
            sfs.time.wfs.plane_25d(array.x, array.n, npw)
        d = sfs.time.wfs.driving_signals(delays, weights, signal)
        plot(d, selection, secondary_source)

    """
    if c is None:
        c = default.c
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    n = util.normalize_vector(n)
    xref = util.asarray_1d(xref)
    g0 = np.sqrt(2 * np.pi * np.linalg.norm(xref - x0, axis=1))
    delays = inner1d(n, x0) / c
    weights = 2 * g0 * inner1d(n, n0)
    selection = util.source_selection_plane(n0, n)
    return delays, weights, selection, secondary_source_point(c)


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
        single secondary source.  See `sfs.time.synthesize()`.

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

    References
    ----------
    See http://sfstoolbox.org/en/latest/#equation-d.wfs.ps.2.5D

    Examples
    --------
    .. plot::
        :context: close-figs

        delays, weights, selection, secondary_source = \
            sfs.time.wfs.point_25d(array.x, array.n, xs)
        d = sfs.time.wfs.driving_signals(delays, weights, signal)
        plot(d, selection, secondary_source, t=ts)

    """
    if c is None:
        c = default.c
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    xs = util.asarray_1d(xs)
    xref = util.asarray_1d(xref)
    g0 = np.sqrt(2 * np.pi * np.linalg.norm(xref - x0, axis=1))
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    delays = r/c
    weights = g0 * inner1d(ds, n0) / (2 * np.pi * r**(3/2))
    selection = util.source_selection_point(n0, x0, xs)
    return delays, weights, selection, secondary_source_point(c)


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
        single secondary source.  See `sfs.time.synthesize()`.

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

    References
    ----------
    See http://sfstoolbox.org/en/latest/#equation-d.wfs.fs.2.5D

    Examples
    --------
    .. plot::
        :context: close-figs

        delays, weights, selection, secondary_source = \
            sfs.time.wfs.focused_25d(array.x, array.n, xf, nf)
        d = sfs.time.wfs.driving_signals(delays, weights, signal)
        plot(d, selection, secondary_source, t=tf)

    """
    if c is None:
        c = default.c
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    xs = util.asarray_1d(xs)
    xref = util.asarray_1d(xref)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    g0 = np.sqrt(np.linalg.norm(xref - x0, axis=1)
                 / (np.linalg.norm(xref - x0, axis=1) + r))
    delays = -r/c
    weights = g0 * inner1d(ds, n0) / (2 * np.pi * r**(3/2))
    selection = util.source_selection_focused(ns, x0, xs)
    return delays, weights, selection, secondary_source_point(c)


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
    delays = util.asarray_1d(delays)
    weights = util.asarray_1d(weights)
    data, samplerate, signal_offset = apply_delays(signal, delays)
    return util.DelayedSignal(data * weights, samplerate, signal_offset)
