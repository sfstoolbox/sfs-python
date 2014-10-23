"""Weights (tapering) for the driving function"""

from scipy import signal


def weight(N): 
    """Weights for the driving function"""
    
    twin = signal.kaiser(N, 2)

    return twin
