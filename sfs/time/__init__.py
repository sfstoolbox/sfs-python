"""Submodules for broadband sound fields.

.. autosummary::
    :toctree:

    nfchoa
    wfs

    source

"""
from . import nfchoa
from . import wfs

from . import source

from .. import util as _util
from .. import array as _array


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
