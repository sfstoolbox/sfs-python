"""Computation of synthesized sound field"""

import sfs
import numpy as np

def generic(x, y, x0, k, d, twin):
    """Compute sound field for a generic driving function"""
    
    p = 0
    weight = d * twin
    
    for n in np.arange(x0.shape[1]):
        p = p +  weight[n] * sfs.source.point(k, x0[:,n], x, y)
        
        
    return p