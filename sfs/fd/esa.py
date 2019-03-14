"""Compute ESA driving functions for various systems.

ESA is abbreviation for equivalent scattering approach.

ESA driving functions for an edge-shaped SSD are provided below.
Further ESA for different geometries might be added here.

Note that mode-matching (such as NFC-HOA, SDM) are equivalent
to ESA in their specific geometries (spherical/circular, planar/linear).

"""

import numpy as np
from scipy.special import jn, hankel2
from .. import util
from . import secondary_source_line, secondary_source_point


def plane_2d_edge(omega, x0, n=[0, 1, 0], *, alpha=3/2*np.pi, Nc=None, c=None):
    r"""Driving function for 2-dimensional plane wave with edge ESA.

    Driving function for a virtual plane wave using the 2-dimensional ESA
    for an edge-shaped secondary source distribution consisting of
    monopole line sources.

    Parameters
    ----------
    omega : float
        Angular frequency.
    x0 : int(N, 3) array_like
        Sequence of secondary source positions.
    n : (3,) array_like, optional
        Normal vector of synthesized plane wave.
    alpha : float, optional
        Outer angle of edge.
    Nc : int, optional
        Number of elements for series expansion of driving function. Estimated
        if not given.
    c : float, optional
        Speed of sound

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.fd.synthesize()`.

    Notes
    -----
    One leg of the secondary sources has to be located on the x-axis (y0=0),
    the edge at the origin.

    Derived from :cite:`Spors2016`

    """
    x0 = np.asarray(x0)
    n = util.normalize_vector(n)
    k = util.wavenumber(omega, c)
    phi_s = np.arctan2(n[1], n[0]) + np.pi
    L = x0.shape[0]

    r = np.linalg.norm(x0, axis=1)
    phi = np.arctan2(x0[:, 1], x0[:, 0])
    phi = np.where(phi < 0, phi+2*np.pi, phi)

    if Nc is None:
        Nc = np.ceil(2 * k * np.max(r) * alpha/np.pi)

    epsilon = np.ones(Nc)  # weights for series expansion
    epsilon[0] = 2

    d = np.zeros(L, dtype=complex)
    for m in np.arange(Nc):
        nu = m*np.pi/alpha
        d = d + 1/epsilon[m] * np.exp(1j*nu*np.pi/2) * np.sin(nu*phi_s) \
            * np.cos(nu*phi) * nu/r * jn(nu, k*r)

    d[phi > 0] = -d[phi > 0]

    selection = util.source_selection_all(len(x0))
    return 4*np.pi/alpha * d, selection, secondary_source_line(omega, c)


def plane_2d_edge_dipole_ssd(omega, x0, n=[0, 1, 0], *, alpha=3/2*np.pi,
                             Nc=None, c=None):
    r"""Driving function for 2-dimensional plane wave with edge dipole ESA.

    Driving function for a virtual plane wave using the 2-dimensional ESA
    for an edge-shaped secondary source distribution consisting of
    dipole line sources.

    Parameters
    ----------
    omega : float
        Angular frequency.
    x0 : int(N, 3) array_like
        Sequence of secondary source positions.
    n : (3,) array_like, optional
        Normal vector of synthesized plane wave.
    alpha : float, optional
        Outer angle of edge.
    Nc : int, optional
        Number of elements for series expansion of driving function. Estimated
        if not given.
    c : float, optional
        Speed of sound

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.fd.synthesize()`.

    Notes
    -----
    One leg of the secondary sources has to be located on the x-axis (y0=0),
    the edge at the origin.

    Derived from :cite:`Spors2016`

    """
    x0 = np.asarray(x0)
    n = util.normalize_vector(n)
    k = util.wavenumber(omega, c)
    phi_s = np.arctan2(n[1], n[0]) + np.pi
    L = x0.shape[0]

    r = np.linalg.norm(x0, axis=1)
    phi = np.arctan2(x0[:, 1], x0[:, 0])
    phi = np.where(phi < 0, phi+2*np.pi, phi)

    if Nc is None:
        Nc = np.ceil(2 * k * np.max(r) * alpha/np.pi)

    epsilon = np.ones(Nc)  # weights for series expansion
    epsilon[0] = 2

    d = np.zeros(L, dtype=complex)
    for m in np.arange(Nc):
        nu = m*np.pi/alpha
        d = d + 1/epsilon[m] * np.exp(1j*nu*np.pi/2) * np.cos(nu*phi_s) \
            * np.cos(nu*phi) * jn(nu, k*r)

    return 4*np.pi/alpha * d


def line_2d_edge(omega, x0, xs, *, alpha=3/2*np.pi, Nc=None, c=None):
    r"""Driving function for 2-dimensional line source with edge ESA.

    Driving function for a virtual line source using the 2-dimensional ESA
    for an edge-shaped secondary source distribution consisting of line
    sources.

    Parameters
    ----------
    omega : float
        Angular frequency.
    x0 : int(N, 3) array_like
        Sequence of secondary source positions.
    xs : (3,) array_like
        Position of synthesized line source.
    alpha : float, optional
        Outer angle of edge.
    Nc : int, optional
        Number of elements for series expansion of driving function. Estimated
        if not given.
    c : float, optional
        Speed of sound

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.fd.synthesize()`.

    Notes
    -----
    One leg of the secondary sources has to be located on the x-axis (y0=0),
    the edge at the origin.

    Derived from :cite:`Spors2016`

    """
    x0 = np.asarray(x0)
    k = util.wavenumber(omega, c)
    phi_s = np.arctan2(xs[1], xs[0])
    if phi_s < 0:
        phi_s = phi_s + 2*np.pi
    r_s = np.linalg.norm(xs)
    L = x0.shape[0]

    r = np.linalg.norm(x0, axis=1)
    phi = np.arctan2(x0[:, 1], x0[:, 0])
    phi = np.where(phi < 0, phi+2*np.pi, phi)

    if Nc is None:
        Nc = np.ceil(2 * k * np.max(r) * alpha/np.pi)

    epsilon = np.ones(Nc)  # weights for series expansion
    epsilon[0] = 2

    d = np.zeros(L, dtype=complex)
    idx = (r <= r_s)
    for m in np.arange(Nc):
        nu = m*np.pi/alpha
        f = 1/epsilon[m] * np.sin(nu*phi_s) * np.cos(nu*phi) * nu/r
        d[idx] = d[idx] + f[idx] * jn(nu, k*r[idx]) * hankel2(nu, k*r_s)
        d[~idx] = d[~idx] + f[~idx] * jn(nu, k*r_s) * hankel2(nu, k*r[~idx])

    d[phi > 0] = -d[phi > 0]

    selection = util.source_selection_all(len(x0))
    return -1j*np.pi/alpha * d, selection, secondary_source_line(omega, c)


def line_2d_edge_dipole_ssd(omega, x0, xs, *, alpha=3/2*np.pi, Nc=None,
                            c=None):
    r"""Driving function for 2-dimensional line source with edge dipole ESA.

    Driving function for a virtual line source using the 2-dimensional ESA
    for an edge-shaped secondary source distribution consisting of dipole line
    sources.

    Parameters
    ----------
    omega : float
        Angular frequency.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    xs : (3,) array_like
        Position of synthesized line source.
    alpha : float, optional
        Outer angle of edge.
    Nc : int, optional
        Number of elements for series expansion of driving function. Estimated
        if not given.
    c : float, optional
        Speed of sound

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.fd.synthesize()`.

    Notes
    -----
    One leg of the secondary sources has to be located on the x-axis (y0=0),
    the edge at the origin.

    Derived from :cite:`Spors2016`

    """
    x0 = np.asarray(x0)
    k = util.wavenumber(omega, c)
    phi_s = np.arctan2(xs[1], xs[0])
    if phi_s < 0:
        phi_s = phi_s + 2*np.pi
    r_s = np.linalg.norm(xs)
    L = x0.shape[0]

    r = np.linalg.norm(x0, axis=1)
    phi = np.arctan2(x0[:, 1], x0[:, 0])
    phi = np.where(phi < 0, phi+2*np.pi, phi)

    if Nc is None:
        Nc = np.ceil(2 * k * np.max(r) * alpha/np.pi)

    epsilon = np.ones(Nc)  # weights for series expansion
    epsilon[0] = 2

    d = np.zeros(L, dtype=complex)
    idx = (r <= r_s)
    for m in np.arange(Nc):
        nu = m*np.pi/alpha
        f = 1/epsilon[m] * np.cos(nu*phi_s) * np.cos(nu*phi)
        d[idx] = d[idx] + f[idx] * jn(nu, k*r[idx]) * hankel2(nu, k*r_s)
        d[~idx] = d[~idx] + f[~idx] * jn(nu, k*r_s) * hankel2(nu, k*r[~idx])

    return -1j*np.pi/alpha * d


def point_25d_edge(omega, x0, xs, *, xref=[2, -2, 0], alpha=3/2*np.pi,
                   Nc=None, c=None):
    r"""Driving function for 2.5-dimensional point source with edge ESA.

    Driving function for a virtual point source using the 2.5-dimensional
    ESA for an edge-shaped secondary source distribution consisting of point
    sources.

    Parameters
    ----------
    omega : float
        Angular frequency.
    x0 : int(N, 3) array_like
        Sequence of secondary source positions.
    xs : (3,) array_like
        Position of synthesized line source.
    xref: (3,) array_like or float
        Reference position or reference distance
    alpha : float, optional
        Outer angle of edge.
    Nc : int, optional
        Number of elements for series expansion of driving function. Estimated
        if not given.
    c : float, optional
        Speed of sound

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.fd.synthesize()`.

    Notes
    -----
    One leg of the secondary sources has to be located on the x-axis (y0=0),
    the edge at the origin.

    Derived from :cite:`Spors2016`

    """
    x0 = np.asarray(x0)
    xs = np.asarray(xs)
    xref = np.asarray(xref)

    if np.isscalar(xref):
        a = np.linalg.norm(xref)/np.linalg.norm(xref-xs)
    else:
        a = np.linalg.norm(xref-x0, axis=1)/np.linalg.norm(xref-xs)

    d, selection, _ = line_2d_edge(omega, x0, xs, alpha=alpha, Nc=Nc, c=c)
    return 1j*np.sqrt(a) * d, selection, secondary_source_point(omega, c)
