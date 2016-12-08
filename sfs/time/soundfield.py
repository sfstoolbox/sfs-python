"""Compute sound field."""

from __future__ import division
import numpy as np
from .. import util
from .. import defs
from .source import point


def p_array(x0, d, channel_weights, t, grid, source=point, fs=None, c=None):
    """Compute sound field for an array of secondary sources.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    d : (N, C) array_like
        Specifies the signals (with N samples) fed into each secondary
        source channel C (columns).
    channel_weights : (C,) array_like
        Additional weights applied during integration, e.g. source
        tapering.
    t : float
        Simulation point in time (seconds).
    grid : triple of array_like
        The grid that is used for the sound field calculations.
        See `sfs.util.xyz_grid()`.
    source: function, optional
        Source type is a function, returning scalar field.
        For default, see `sfs.time.source.point()`.
    fs: int, optional
        Sampling frequency in Hertz.
    c : float, optional
        Speed of sound.

    Returns
    -------
    numpy.ndarray
        Sound pressure at grid positions.

    """
    if fs is None:
        fs = defs.fs
    if c is None:
        c = defs.c
    x0 = util.asarray_of_rows(x0)
    channel_weights = util.asarray_1d(channel_weights)
    d = np.asarray(d)
    if not (len(channel_weights) == len(x0) == d.shape[1]):
        raise ValueError("Length mismatch")
    # synthesize soundfield
    p = 0
    for signal, weight, position in zip(d.T, channel_weights, x0):
        if weight != 0:
            p_s = source(position, signal, t, grid, fs, c)
            p += p_s * weight  # integrate over secondary sources
    return p
