"""Weights (tapering) for the driving function"""

# from scipy import signal
import numpy as np


def none(active): 
    """No tapering window."""
    return np.asarray(active, dtype=np.float64)
    
def kaiser(active):
    """Kaiser tapering window."""
    active = np.asarray(active, dtype=np.float64)
    idx = np.where(active == 1)[0]
    print(active)
    window = np.zeros(active.shape)
    window[idx] = np.kaiser(idx.shape[0], 2)
    return window