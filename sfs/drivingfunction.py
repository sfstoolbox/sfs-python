"""Compute driving functions for various systems"""

import numpy as np


def pw_delay(k, x0, n0, npw=[0, 1, 0]):
    """Plane wave by simple delay of secondary sources."""
    x0 = np.asarray(x0)
    npw = np.squeeze(np.asarray(npw))
    return np.exp(-1j * k * np.inner(npw, x0))
    
def pw_3D_WFS(k, x0, n0, npw=[0, 1, 0]):
    """Plane wave by three dimensional WFS. Eq.(17) from [Spors et al, 2008]"""
    x0 = np.asarray(x0)
    npw = np.squeeze(np.asarray(npw))
    return -2j*k*np.inner(npw,n0)*np.exp(-1j * k * np.inner(npw, x0))
    
    
def source_selection_pw(n0,npw):
    """Secondary source selection for a plane wave. Eq.(13) from [Spors et al, 2008]"""
    selected = np.inner(npw,n0)    
    return selected>=0