"""Weights (tapering) for the driving function."""

# from scipy import signal
import numpy as np


def none(active):
    """No tapering window."""
    return np.asarray(active, dtype=np.float64)


def kaiser(active):
    """Kaiser tapering window."""
    # TODO: window for closed arrays not working
    active = np.asarray(active, dtype=np.float64)
    idx = np.flatnonzero(active)
    window = np.zeros(active.shape)
    window[idx] = np.kaiser(len(idx), 2)
    return window
