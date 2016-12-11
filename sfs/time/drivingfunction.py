"""Compute time based driving functions for various systems.

.. include:: math-definitions.rst

"""
from __future__ import division
import numpy as np
from numpy.core.umath_tests import inner1d  # element-wise inner product
from .. import defs
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

    """
    if c is None:
        c = defs.c
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    n = util.asarray_1d(n)
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

    """
    if c is None:
        c = defs.c
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


def driving_signals(delays, weights, signal, fs=None):
    """Get driving signals per secondary source.

    Returned signals are the delayed and weighted mono input signal
    (with N samples) per channel (C).

    Parameters
    ----------
    delays : (C,) array_like
        Delay in seconds for each channel, negative values allowed.
    weights : (C,) array_like
        Amplitude weighting factor for each channel.
    signal : (N,) array_like
        Excitation signal (mono) which gets weighted and delayed.
    fs: int, optional
        Sampling frequency in Hertz.

    Returns
    -------
    driving_signals : (N, C) numpy.ndarray
        Driving signal per channel (column represents channel).
    t_offset : float
        Simulation point in time offset (seconds).

    """
    delays = util.asarray_1d(delays)
    weights = util.asarray_1d(weights)
    d, t_offset = apply_delays(signal, delays, fs)
    return d * weights, t_offset


def apply_delays(signal, delays, fs=None):
    """Apply delays for every channel.

    A mono input signal gets delayed for each channel individually. The
    simultation point in time is shifted by the smallest delay provided,
    which allows negative delays as well.

    Parameters
    ----------
    signal : (N,) array_like
        Mono excitation signal (with N samples) which gets delayed.
    delays : (C,) array_like
        Delay in seconds for each channel (C), negative values allowed.
    fs: int, optional
        Sampling frequency in Hertz.

    Returns
    -------
    out : (N, C) numpy.ndarray
        Output signals (column represents channel).
    t_offset : float
        Simulation point in time offset (seconds).

    """
    if fs is None:
        fs = defs.fs
    signal = util.asarray_1d(signal)
    delays = util.asarray_1d(delays)

    delays_samples = np.rint(fs * delays).astype(int)
    offset_samples = delays_samples.min()
    delays_samples -= offset_samples
    out = np.zeros((delays_samples.max() + len(signal), len(delays_samples)))
    for channel, cdelay in enumerate(delays_samples):
        out[cdelay:cdelay + len(signal), channel] = signal
    return out, offset_samples / fs


def wfs_prefilter(dim='2.5D', N=128, fl=50, fu=2000, fs=None, c=None):
    """Get pre-equalization filter for WFS.

    Rising slope with 3dB/oct ('2.5D') or 6dB/oct ('2D' and '3D').
    Constant magnitude below fl and above fu.
    Type 1 linear phase FIR filter of order N.
    Simple design via "frequency sampling method".

    Parameters
    ----------
    N : int, optional
        Filter order, shall be even. For odd N, N+1 is used.
    dim : str, optional
        Dimensionality, must be '2D', '2.5D' or '3D'.
    fl : int, optional
        Lower corner frequency in Hertz.
    fu : int, optional
        Upper corner frequency in Hertz.
        (Should be around spatial aliasing limit.)
    fs : int, optional
        Sampling frequency in Hertz.
    c : float, optional
        Speed of sound.

    Returns
    -------
    h : (N+1,) numpy.ndarray
        Filter taps.
    delay : float
        Pre-delay in seconds.

    """
    N = 2*(int(N + 1)//2)  # for odd N, use N+1 instead
    if fs is None:
        fs = defs.fs
    if c is None:
        c = defs.c

    numbins = int(N/2 + 1)
    delta_f = fs / (2*numbins - 1)
    f = np.arange(numbins) * delta_f
    if dim == '2D' or dim == '3D':
        alpha = 1
    elif dim == '2.5D':
        alpha = 0.5

    desired = np.power(2 * np.pi * f / c, alpha)
    low_shelf = np.power(2 * np.pi * fl / c, alpha)
    high_shelf = np.power(2 * np.pi * fu / c, alpha)

    l_index = int(np.ceil(fl / delta_f))
    u_index = int(min(np.ceil(fu / delta_f), numbins - 1))
    desired[:l_index] = low_shelf
    desired[u_index:] = min(high_shelf, desired[u_index])

    h = np.fft.ifft(np.concatenate((desired, desired[-1:0:-1])))
    h = np.roll(np.real(h), numbins - 1)
    delay = (numbins - 1) / fs
    return h, delay
