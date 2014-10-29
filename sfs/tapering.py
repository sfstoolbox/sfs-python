"""Weights (tapering) for the driving function"""

# from scipy import signal
import numpy as np


def none(active): 
    """No tapering window."""
    return np.asarray(active, dtype=np.float64)
