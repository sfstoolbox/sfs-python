"""Compute sound field."""

from __future__ import division
import numpy as np
from .. import util
from .. import defs
from .source import point


def p_array(x0, d, channel_weights, t, grid, source=point, fs=None, c=None):
    """Compute sound field for an array of secondary sources for a given time.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    d : (N, C) numpy.ndarray
        Specifies the signals N(t) fed into each secondary source C (columns).
        "N" is signal samples(t) per channel, and
        "C" is channels
    channel_weights : (C,) array_like
        Additional weights applied during integration, e.g. source tapering.
    t : float
        Simulation point in time (seconds).
    grid : triple of array_like
        The grid that is used for the sound field calculations.
        See `sfs.util.xyz_grid()`.
    source: function, optional
        Source type is a function, returning scalar field.
        For default, see `sfs.time.source.point()`
    fs: int, optional
        Sampling frequency in hertz.
    c : float, optional
        Speed of sound.

    Returns
    -------
    numpy.ndarray
        Pressure at grid positions.

    """
    if fs is None:
        fs = defs.fs
    if c is None:
        c = defs.c
    x0 = util.asarray_of_rows(x0)
    channel_weights = util.asarray_1d(channel_weights)
    if np.size(channel_weights, 0) != np.size(x0, 0):
        raise ValueError("length mismatch")

    # synthesize soundfield
    p = 0
    for signal, weight, position in zip(d.T, channel_weights, x0):
        if weight != 0:
            p_s = source(position, signal, t, grid, fs, c)
            p += p_s * weight  # integrate over secondary sources
    return p
