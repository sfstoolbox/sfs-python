"""Compute driving functions for various systems"""

import numpy as np


def wfs_ps(k, x0, n0, xs):
    """Point source by two-dimensional WFS."""
    #                (x0-xs) n0
    # D(x0,k) = j k ------------- e^(-j k |x0-xs|)
    #               |x0-xs|^(3/2)
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    xs = np.asarray(xs)
    r = x0-xs
    return 1j * k * np.inner(r, n0) / np.abs(r) ** (3 / 2) * \
        np.exp(-1j * k * np.abs(r))


def wfs_2d_ps(k, x0, n0, xs):
    """Point source by two-dimensional WFS."""
    return wfs_ps(k, x0, n0, xs)


def wfs_25d_ps(k, x0, n0, xs, xref=[0, 0, 0]):
    """Point source by 2.5-dimensional WFS."""
    #             ____________   (x0-xs) n0
    # D(x0,k) = \|j k |xref-x0| ------------- e^(-j k |x0-xs|)
    #                           |x0-xs|^(3/2)
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    xs = np.asarray(xs)
    xref = np.asarray(xref)
    r = x0-xs
    return np.sqrt(1j * k * np.abs(xref - x0)) * np.inner(r, n0) / \
        np.abs(r) ** (3 / 2) * np.exp(-1j * k * np.abs(r))


def wfs_3d_ps(k, x0, nx0, xs):
    """Point source by three-dimensional WFS."""
    return wfs_ps(k, x0, nx0, xs)


def wfs_pw(k, x0, n0, n=[0, 1, 0]):
    """Plane wave by WFS."""
    # D(x0,k) =  j k n n0  e^(-j k n x0)
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    n = np.squeeze(np.asarray(n))
    return 1j * k * np.inner(n, n0) * np.exp(-1j * k * np.inner(n, n0))


def wfs_2d_pw(k, x0, n0, n=[0, 1, 0]):
    """Plane wave by two-dimensional WFS."""
    return wfs_pw(k, x0, n0, n)


def wfs_25d_pw(k, x0, n0, n=[0, 1, 0], xref=[0, 0, 0]):
    """Plane wave by 2.5-dimensional WFS."""
    #                  ____________
    # D_2.5D(x0,w) = \|j k |xref-x0| n n0 e^(-j k n x0)
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    n = np.squeeze(np.asarray(n))
    return np.sqrt(1j * k * np.abs(xref-x0)) * np.inner(n, n0) * \
        np.exp(-1j * k * np.inner(n, x0))


def wfs_3d_pw(k, x0, n0, n=[0, 1, 0]):
    """Plane wave by three-dimensional WFS.
       Eq.(17) from [Spors et al, 2008]"""
    return wfs_pw(k, x0, n0, n)


def delay_3d_pw(k, x0, n0, n=[0, 1, 0]):
    """Plane wave by simple delay of secondary sources."""
    x0 = np.asarray(x0)
    n = np.squeeze(np.asarray(n))
    return np.exp(-1j * k * np.inner(n, x0))


def source_selection_pw(n0, n):
    """Secondary source selection for a plane wave.
       Eq.(13) from [Spors et al, 2008]"""
    n0 = np.asarray(n0)
    n = np.asarray(n)
    return np.inner(n, n0) >= 0
