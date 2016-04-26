"""Compute sound field."""

import numpy as np

from .. import util
from . import source
from .. import defs


def synthesize_p(d_line, channel_weight, x0, grid, t, fs=None):
    """Compute sound field for given time sample.

    D_line(m x n) specifies the signals m fed into each secondary source n.
    Channel_weight is applied during integration, e.g. source tapering.
    x0 are available secandary sources.

    """
    if fs is None:
        fs = defs.fs
    channel_weight = util.asarray_1d(channel_weight)
    if len(channel_weight) != len(x0):
        raise ValueError("length mismatch")

    delay = np.nonzero(d_line)[0]
    offset = np.min(delay)
    d_line = d_line[offset:, :]  # remove driving function time offset

    channels = len(x0)  # extract no. of channels
    # Add additional zeros to the driving signal to ensure an amplitude of 0
    # in the whole listening area before and after the real driving signal
    padding_samples = 1000  # TODO: round(max(max_distance/c*fs,2*L/c*fs))
    zero_fill = np.zeros([padding_samples, channels])

    # time reversal driving function
    sig_length = len(d_line)
    t_inverted = t - sig_length - padding_samples  # compensate time
    # zero padding and reverse time signals
    d_line_inverted = np.flipud(np.row_stack((zero_fill, d_line, zero_fill)))

    # synthesize soundfield
    p = 0
    for signal, weight, position in zip(d_line_inverted.T, channel_weight, x0):
        if weight != 0:
            g, t_delta = source.greens_function_ps(position, grid, t_inverted)
            d_intpl = np.interp(t_delta, np.arange(np.size(signal)), signal)
            p += g * d_intpl * weight  # integrate

    return p
