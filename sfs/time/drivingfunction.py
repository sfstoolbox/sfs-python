"""Compute time based driving functions for various systems.

.. include:: math-definitions.rst

.. plot::
    :context: reset

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import unit_impulse
    import sfs

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
    signal = unit_impulse(512), 44100

    # Circular loudspeaker array
    N = 32  # number of loudspeakers
    R = 1.5  # radius
    x0, n0, a0 = sfs.array.circular(N, R)

    grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.02)

    def plot(d, selected, t=0):
        p = sfs.time.soundfield.p_array(x0, d, selected * a0, t, grid)
        sfs.plot.level(p, grid)
        sfs.plot.loudspeaker_2d(x0, n0, selected * a0, size=0.15)

"""
import numpy as np
from numpy.core.umath_tests import inner1d  # element-wise inner product
from .. import default
from .. import util


def wfs_25d_plane(x0, n0, n=[0, 1, 0], xref=[0, 0, 0], c=None):
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
    weights: (N,) numpy.ndarray
        Weights of secondary sources.

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

        delays, weights = sfs.time.drivingfunction.wfs_25d_plane(x0, n0, npw)
        d = sfs.time.drivingfunction.driving_signals(delays, weights, signal)
        a = sfs.mono.drivingfunction.source_selection_plane(n0, npw)
        plot(d, a)

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
    return delays, weights


def wfs_25d_point(x0, n0, xs, xref=[0, 0, 0], c=None):
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

        delays, weights = sfs.time.drivingfunction.wfs_25d_point(x0, n0, xs)
        d = sfs.time.drivingfunction.driving_signals(delays, weights, signal)
        a = sfs.mono.drivingfunction.source_selection_point(n0, x0, xs)
        plot(d, a, t=ts)

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
    return delays, weights


def wfs_25d_focused(x0, n0, xs, xref=[0, 0, 0], c=None):
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

        delays, weights = sfs.time.drivingfunction.wfs_25d_focused(x0, n0, xf)
        d = sfs.time.drivingfunction.driving_signals(delays, weights, signal)
        a = sfs.mono.drivingfunction.source_selection_focused(nf, x0, xf)
        plot(d, a, t=tf)

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
    return delays, weights


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


def apply_delays(signal, delays):
    """Apply delays for every channel.

    Parameters
    ----------
    signal : (N,) array_like + float
        Excitation signal consisting of (mono) audio data and a sampling
        rate (in Hertz).  A `DelayedSignal` object can also be used.
    delays : (C,) array_like
        Delay in seconds for each channel (C), negative values allowed.

    Returns
    -------
    `DelayedSignal`
        A tuple containing the delayed signals (in a `numpy.ndarray`
        with shape ``(N, C)``), followed by the sampling rate (in Hertz)
        and a (possibly negative) time offset (in seconds).

    """
    data, samplerate, initial_offset = util.as_delayed_signal(signal)
    data = util.asarray_1d(data)
    delays = util.asarray_1d(delays)
    delays += initial_offset

    delays_samples = np.rint(samplerate * delays).astype(int)
    offset_samples = delays_samples.min()
    delays_samples -= offset_samples
    out = np.zeros((delays_samples.max() + len(data), len(delays_samples)))
    for column, row in enumerate(delays_samples):
        out[row:row + len(data), column] = data
    return util.DelayedSignal(out, samplerate, offset_samples / samplerate)
