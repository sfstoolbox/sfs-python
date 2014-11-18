"""Weights (tapering) for the driving function."""

# from scipy import signal
import numpy as np


def none(active):
    """No tapering window."""
    return np.asarray(active, dtype=np.float64)


def kaiser(active):
    """Kaiser tapering window."""
    idx = _windowidx(active)
    # compute coefficients
    window = np.zeros(active.shape)
    window[idx] = np.kaiser(len(idx), 2)
    return window


def _windowidx(active):
    """Returns list of connected indices for window function."""
    active = np.asarray(active, dtype=np.float64)
    # find index were active loudspeakers begin (works for connected contours)
    if active[0] == 1 and active[-1] == 0:
        a0 = 0
    else:
        a0 = np.argmax(np.diff(active)) + 1
    # shift generic index vector to get a connected list of indices
    idx = np.roll(np.arange(len(active)), -a0)
    # remove indices of inactive secondary sources
    idx = idx[0:len(np.squeeze(np.where(active == 1)))]
    return idx
