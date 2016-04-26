"""Compute time based driving functions for various systems."""

import numpy as np
from numpy.core.umath_tests import inner1d  # element-wise inner product
from .. import defs
from .. import util


def wfs_25d_ps(xs, x0, nx0, xref=[0, 0, 0], fs=None, c=None):
    """Point source by 2.5-dimensional WFS.

    ::

         2.5D correction factor
                ______________
         g0 = \| 2pi |xref-x0|


         d_2.5D using a point source as source model

                               g0  (x0-xs) nx0
         d_2.5D(x0,t) = h(t) * --- ------------- delta(t-|x0-xs|/c)
                               2pi |x0-xs|^(3/2)

         see Wierstorf et al. (2015), eq.(#d:wfs:ps:2.5D)

    """
    if c is None:
        c = defs.c
    if fs is None:
        fs = defs.fs
    x0 = util.asarray_of_rows(x0)
    nx0 = util.asarray_of_rows(nx0)
    xs = util.asarray_1d(xs)
    g0 = np.sqrt(2*np.pi*np.linalg.norm(xref-x0, axis=1))
    ds = x0 - xs
    r = np.linalg.norm(x0-xs, axis=1)
    delay = 1/c * r
    weight = g0/(2*np.pi) * inner1d(ds, nx0) / r**(3/2)
    sig = np.ones(len(x0))  # dirac
    line = delayline(sig, delay*fs, weight)
    return delay, weight, line


def delayline(sig, delay, weight):
    """Delayline with weights, no fractional delay yet.

    Delay in samples.
    sig (m x n) where columns n are channels and rows m the corresponding
    signals.
    Weight is array with weighting for each channel.

    """
    delay = np.rint(delay).astype(int)  # no fractional delay, samples
    extention = np.max(delay)  # avoid cutting signal
    channels = len(weight)  # extract no. of channels
    sig_length = sig.ndim
    out = np.zeros([sig_length+extention, channels])  # create shape

    for channel in range(channels):
        cdelay = delay[channel]  # get channel delay
        out[cdelay:cdelay+sig_length, channel] = sig[channel] * weight[channel]

    return out
