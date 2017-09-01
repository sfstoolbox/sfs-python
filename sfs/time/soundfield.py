"""Compute sound field."""

from __future__ import division
from .. import util
from .. import defs
from .source import point


def p_array(x0, signals, weights, observation_time, grid, source=point,
            c=None):
    """Compute sound field for an array of secondary sources.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    signals : (N, C) array_like + float
        Driving signals consisting of audio data (C channels) and a
        sampling rate (in Hertz).
        A `DelayedSignal` object can also be used.
    weights : (C,) array_like
        Additional weights applied during integration, e.g. source
        tapering.
    observation_time : float
        Simulation point in time (seconds).
    grid : triple of array_like
        The grid that is used for the sound field calculations.
        See `sfs.util.xyz_grid()`.
    source: function, optional
        Source type is a function, returning scalar field.
        For default, see `sfs.time.source.point()`.
    c : float, optional
        Speed of sound.

    Returns
    -------
    numpy.ndarray
        Sound pressure at grid positions.

    """
    if c is None:
        c = defs.c
    x0 = util.asarray_of_rows(x0)
    data, samplerate, signal_offset = util.as_delayed_signal(signals)
    weights = util.asarray_1d(weights)
    channels = data.T
    if not (len(weights) == len(x0) == len(channels)):
        raise ValueError("Length mismatch")
    # synthesize soundfield
    p = 0
    for channel, weight, position in zip(channels, weights, x0):
        if weight != 0:
            signal = channel, samplerate, signal_offset
            p_s = source(position, signal, observation_time, grid, c)
            p += p_s * weight  # integrate over secondary sources
    return p
