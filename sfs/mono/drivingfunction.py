"""Compute driving functions for various systems."""

import numpy as np
from numpy.core.umath_tests import inner1d  # element-wise inner product
from scipy.special import hankel2
from scipy.special import sph_jn, sph_yn
from .. import util
from .. import defs


def wfs_2d_line(omega, x0, n0, xs, c=None):
    """Line source by 2-dimensional WFS.

    ::

        D(x0,k) = j k (x0-xs) n0 / |x0-xs| * H1(k |x0-xs|)

    """
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    xs = np.squeeze(np.asarray(xs))
    k = util.wavenumber(omega, c)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    return 1j * k * inner1d(ds, n0) / r * hankel2(1, k * r)


def _wfs_point(omega, x0, n0, xs, c=None):
    """Point source by two- or three-dimensional WFS.

    ::

                       (x0-xs) n0
        D(x0,k) = j k ------------- e^(-j k |x0-xs|)
                      |x0-xs|^(3/2)

    """
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    xs = np.squeeze(np.asarray(xs))
    k = util.wavenumber(omega, c)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    return 1j * k * inner1d(ds, n0) / r ** (3 / 2) * np.exp(-1j * k * r)


wfs_2d_point = _wfs_point


def wfs_25d_point(omega, x0, n0, xs, xref=[0, 0, 0], c=None, omalias=None):
    """Point source by 2.5-dimensional WFS.

    ::

                    ____________   (x0-xs) n0
        D(x0,k) = \|j k |xref-x0| ------------- e^(-j k |x0-xs|)
                                  |x0-xs|^(3/2)

    """
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    xs = np.squeeze(np.asarray(xs))
    xref = np.squeeze(np.asarray(xref))
    k = util.wavenumber(omega, c)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)

    return wfs_25d_preeq(omega, omalias, c) * \
        np.sqrt(np.linalg.norm(xref - x0)) * inner1d(ds, n0) / \
        r ** (3 / 2) * np.exp(-1j * k * r)


wfs_3d_point = _wfs_point


def _wfs_plane(omega, x0, n0, n=[0, 1, 0], c=None):
    """Plane wave by two- or three-dimensional WFS.

    Eq.(17) from [Spors et al, 2008]::

        D(x0,k) =  j k n n0  e^(-j k n x0)

    """
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    n = np.squeeze(np.asarray(n))
    k = util.wavenumber(omega, c)
    return 2j * k * np.inner(n, n0) * np.exp(-1j * k * np.inner(n, x0))


wfs_2d_plane = _wfs_plane


def wfs_25d_plane(omega, x0, n0, n=[0, 1, 0], xref=[0, 0, 0], c=None,
                  omalias=None):
    """Plane wave by 2.5-dimensional WFS.

    ::

                         ____________
        D_2.5D(x0,w) = \|j k |xref-x0| n n0 e^(-j k n x0)

    """
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    n = np.squeeze(np.asarray(n))
    xref = np.squeeze(np.asarray(xref))
    k = util.wavenumber(omega, c)
    return wfs_25d_preeq(omega, omalias, c) * \
        np.sqrt(2*np.pi * np.linalg.norm(xref - x0)) * \
        np.inner(n, n0) * np.exp(-1j * k * np.inner(n, x0))


wfs_3d_plane = _wfs_plane


def wfs_25d_preeq(omega, omalias, c):
    """Preqeualization for 2.5D WFS."""
    if omalias is None:
        return np.sqrt(1j * util.wavenumber(omega, c))
    else:
        if omega <= omalias:
            return np.sqrt(1j * util.wavenumber(omega, c))
        else:
            return np.sqrt(1j * util.wavenumber(omalias, c))


def delay_3d_plane(omega, x0, n0, n=[0, 1, 0], c=None):
    """Plane wave by simple delay of secondary sources."""
    x0 = np.asarray(x0)
    n = np.squeeze(np.asarray(n))
    k = util.wavenumber(omega, c)
    return np.exp(-1j * k * np.inner(n, x0))


def source_selection_plane(n0, n):
    """Secondary source selection for a plane wave.

    Eq.(13) from [Spors et al, 2008]

    """
    n0 = np.asarray(n0)
    n = np.squeeze(np.asarray(n))
    return np.inner(n, n0) >= defs.selection_tolerance


def source_selection_point(n0, x0, xs):
    """Secondary source selection for a point source.

    Eq.(15) from [Spors et al, 2008]

    """
    n0 = np.asarray(n0)
    x0 = np.asarray(x0)
    xs = np.squeeze(np.asarray(xs))
    ds = x0 - xs
    return inner1d(ds, n0) >= defs.selection_tolerance


def source_selection_all(N):
    """Select all secondary sources."""
    return np.ones(N) >= 0


def nfchoa_2d_plane(omega, x0, r0, n=[0, 1, 0], c=None):
    """Point source by 2.5-dimensional WFS."""
    x0 = np.asarray(x0)
    k = util.wavenumber(omega, c)
    alpha, beta, r = util.cart2sph(n[0], n[1], n[2])
    alpha0, beta0, tmp = util.cart2sph(x0[:, 0], x0[:, 1], x0[:, 2])
    # determine max order of circular harmonics
    M = _hoa_order_2d(len(x0))
    # compute driving function
    d = 0
    for m in np.arange(-M, M):
        d = d + 1j**(-m) / hankel2(m, k * r0) * \
            np.exp(1j * m * (alpha0 - alpha))

    return - 2j / (np.pi*r0) * d


def nfchoa_25d_point(omega, x0, r0, xs, c=None):
    """Point source by 2.5-dimensional WFS.

    ::

                             __      (2)
                      1     \       h|m| (w/c r)
        D(phi0,w) = -----   /__    ------------- e^(i m (phi0-phi))
                     2pi r0 m=-N..N  (2)
                                    h|m| (w/c r0)

    """
    x0 = np.asarray(x0)
    k = util.wavenumber(omega, c)
    alpha, beta, r = util.cart2sph(xs[0], xs[1], xs[2])
    alpha0, beta0, tmp = util.cart2sph(x0[:, 0], x0[:, 1], x0[:, 2])
    # determine max order of circular harmonics
    M = _hoa_order_2d(len(x0))
    # compute driving function
    d = 0
    a = _sph_hn2(M, k * r) / _sph_hn2(M, k * r0)
    for m in np.arange(-M, M):
        d += a[0, abs(m)] * np.exp(1j * m * (alpha0 - alpha))

    return 1 / (2 * np.pi * r0) * d


def nfchoa_25d_plane(omega, x0, r0, n=[0, 1, 0], c=None):
    """Plane wave by 2.5-dimensional WFS.

    ::

                             __
                        2i  \            i^|m|
        D_25D(phi0,w) = --  /__    ------------------ e^(i m (phi0-phi_pw) )
                        r0 m=-N..N       (2)
                                    w/c h|m| (w/c r0)

    """
    x0 = np.asarray(x0)
    k = util.wavenumber(omega, c)
    alpha, beta, r = util.cart2sph(n[0], n[1], n[2])
    alpha0, beta0, tmp = util.cart2sph(x0[:, 0], x0[:, 1], x0[:, 2])
    # determine max order of circular harmonics
    M = _hoa_order_2d(len(x0))
    # compute driving function
    d = 0
    a = 1 / _sph_hn2(M, k * r0)
    for m in np.arange(-M, M):
        d += (1j)**(-abs(m)) * a[0, abs(m)] * \
            np.exp(1j * m * (alpha0 - alpha))

    return -2 / r0 * d


def sdm_2d_line(omega, x0, n0, xs, c=None):
    """Line source by two-dimensional SDM.

    The secondary sources have to be located on the x-axis (y0=0).
    Derived from [Spors 2009, 126th AES Convention], Eq.(9), Eq.(4)::

        D(x0,k) =

    """
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    xs = np.squeeze(np.asarray(xs))
    k = util.wavenumber(omega, c)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    return - 1j/2 * k * xs[1] / r * hankel2(1, k * r)


def sdm_2d_plane(omega, x0, n0, n=[0, 1, 0], c=None):
    """Plane wave by two-dimensional SDM.

    The secondary sources have to be located on the x-axis (y0=0).
    Derived from [Ahrens 2011, Springer], Eq.(3.73), Eq.(C.5), Eq.(C.11)::

        D(x0,k) = kpw,y * e^(-j*kpw,x*x)

    """
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    n = np.squeeze(np.asarray(n))
    k = util.wavenumber(omega, c)
    return k * n[1] * np.exp(-1j * k * n[0] * x0[:, 0])


def sdm_25d_plane(omega, x0, n0, n=[0, 1, 0], xref=[0, 0, 0], c=None):
    """Plane wave by 2.5-dimensional SDM.

    The secondary sources have to be located on the x-axis (y0=0).
    Eq.(3.79) from [Ahrens 2011, Springer]::

        D_2.5D(x0,w) =

    """
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    n = np.squeeze(np.asarray(n))
    xref = np.squeeze(np.asarray(xref))
    k = util.wavenumber(omega, c)
    return 4j * np.exp(-1j*k*n[1]*xref[1]) / hankel2(0, k*n[1]*xref[1]) * \
        np.exp(-1j*k*n[0]*x0[:, 0])


def sdm_25d_point(omega, x0, n0, xs, xref=[0, 0, 0], c=None):
    """Point source by 2.5-dimensional SDM.

    The secondary sources have to be located on the x-axis (y0=0).
    Driving funcnction from [Spors 2010, 128th AES Covention], Eq.(24)::

        D(x0,k) =

    """
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    xs = np.squeeze(np.asarray(xs))
    xref = np.squeeze(np.asarray(xref))
    k = util.wavenumber(omega, c)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    return 1/2 * 1j * k * np.sqrt(xref[1] / (xref[1] - xs[1])) * \
        xs[1] / r * hankel2(1, k * r)


def _sph_hn2(n, z):
    """Spherical Hankel function of 2nd kind."""
    return np.asarray(sph_jn(n, z)) - 1j * np.asarray(sph_yn(n, z))


def _hoa_order_2d(N):
    """Computes order of HOA."""
    if N % 2 == 0:
        return N//2
    else:
        return (N-1)//2
