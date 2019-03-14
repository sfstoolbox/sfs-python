"""Submodules for broadband sound fields.

.. autosummary::
    :toctree:

    source

    wfs
    nfchoa

"""
from .. import array as _array
import numpy as np
from . import source
from .. import util as _util


def synthesize(signals, weights, ssd, secondary_source_function, **kwargs):
    """Compute sound field for an array of secondary sources.

    Parameters
    ----------
    signals : (N, C) array_like + float
        Driving signals consisting of audio data (C channels) and a
        sampling rate (in Hertz).
        A `DelayedSignal` object can also be used.
    weights : (C,) array_like
        Additional weights applied during integration, e.g. source
        selection and tapering.
    ssd : sequence of between 1 and 3 array_like objects
        Positions (shape ``(C, 3)``), normal vectors (shape ``(C, 3)``)
        and weights (shape ``(C,)``) of secondary sources.
        A `SecondarySourceDistribution` can also be used.
    secondary_source_function : callable
        A function that generates the sound field of a secondary source.
        This signature is expected::

            secondary_source_function(
                position, normal_vector, weight, driving_signal,
                **kwargs) -> numpy.ndarray

    **kwargs
        All keyword arguments are forwarded to *secondary_source_function*.
        This is typically used to pass the *observation_time* and *grid*
        arguments.

    Returns
    -------
    numpy.ndarray
        Sound pressure at grid positions.

    """
    ssd = _array.as_secondary_source_distribution(ssd)
    data, samplerate, signal_offset = _util.as_delayed_signal(signals)
    weights = _util.asarray_1d(weights)
    channels = data.T
    if not (len(ssd.x) == len(ssd.n) == len(ssd.a) == len(channels) ==
            len(weights)):
        raise ValueError("Length mismatch")
    p = 0
    for x, n, a, channel, weight in zip(ssd.x, ssd.n, ssd.a,
                                        channels, weights):
        if weight != 0:
            signal = channel, samplerate, signal_offset
            p += a * weight * secondary_source_function(x, n, signal, **kwargs)
    return p


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
    data, samplerate, initial_offset = _util.as_delayed_signal(signal)
    data = _util.asarray_1d(data)
    delays = _util.asarray_1d(delays)
    delays += initial_offset

    delays_samples = np.rint(samplerate * delays).astype(int)
    offset_samples = delays_samples.min()
    delays_samples -= offset_samples
    out = np.zeros((delays_samples.max() + len(data), len(delays_samples)))
    for column, row in enumerate(delays_samples):
        out[row:row + len(data), column] = data
    return _util.DelayedSignal(out, samplerate, offset_samples / samplerate)


def secondary_source_point(c):
    """Create a point source for use in `sfs.td.synthesize()`."""

    def secondary_source(position, _, signal, observation_time, grid):
        return source.point(position, signal, observation_time, grid, c=c)

    return secondary_source


from . import nfchoa
from . import wfs
