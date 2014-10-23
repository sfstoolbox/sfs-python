"""Positions of various secondary source distributions"""

import numpy as np


def linear(N,dx,center=[0, 0 ,0]):
    """Linear secondary source distribution parallel to the x-axis"""
    
    # x0=np.zeros((3,N))

    xpos = ( np.arange(N) - N/2 + 1/2 ) * dx + center[0]
    
    x0 = [ xpos , center[1]*np.ones(N) , center[2]*np.ones(N) ]
    
    x0 = np.array(x0)
    
    return x0 