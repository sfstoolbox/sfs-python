"""Compute ESA driving functions for various systems.

ESA is abbreviation for equivalent scattering approach.

ESA driving functions for an edge-shaped SSD are provided below.
Further ESA for different geometries might be added here.

Note that mode-matching (such as NFC-HOA, SDM) are equivalent
to ESA in their specific geometries (spherical/circular, planar/linear).

.. plot::
    :context: reset

    import matplotlib.pyplot as plt
    import numpy as np
    import sfs

    plt.rcParams['figure.figsize'] = 6, 6

    f = 343  # Hz
    omega = 2 * np.pi * f  # rad / s
    k = omega / sfs.default.c  # rad / m

    npw = sfs.util.direction_vector(np.radians(-45))
    xs = np.array([-0.828427, 0.828427, 0])

    grid = sfs.util.xyz_grid([-1, 5], [-5, 1], 0, spacing=0.02)
    dx, L = 0.05, 4  # m
    N = int(L / dx)
    array = sfs.array.edge(N, dx, center=[0, 0, 0],
                           orientation=[0, -1, 0])

    xref = np.array([2, -2, 0])
    x_norm = np.linalg.norm(xs - xref)
    norm_ls = (np.sqrt(8 * np.pi * k * x_norm) *
               np.exp(+1j * np.pi / 4) *
               np.exp(-1j * k * x_norm))
    norm_pw = np.exp(+1j * 4*np.pi*np.sqrt(2))


    def plot(d, selection, secondary_source, norm_ref):
        # the series expansion is numerically tricky, hence
        d = np.nan_to_num(d)
        # especially handle the origin loudspeaker
        d[N] = 0  # as it tends to nan/inf
        p = sfs.fd.synthesize(d, selection, array, secondary_source, grid=grid)
        sfs.plot2d.amplitude(p * norm_ref, grid)
        sfs.plot2d.loudspeakers(array.x, array.n,
                                selection * array.a, size=0.15)
        plt.xlim(-0.5, 4.5)
        plt.ylim(-4.5, 0.5)
        plt.grid(True)

"""
import numpy as _np
from scipy.special import jn as _jn, hankel2 as _hankel2

from . import secondary_source_line as _secondary_source_line
from . import secondary_source_point as _secondary_source_point
from .. import util as _util


def plane_2d_edge(omega, x0, n=[0, 1, 0], *, alpha=_np.pi*3/2, Nc=None,
                  c=None):
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

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.fd.esa.plane_2d_edge(
            omega, array.x, npw, alpha=np.pi*3/2)
        plot(d, selection, secondary_source, norm_pw)

    """
    x0 = _np.asarray(x0)
    n = _util.normalize_vector(n)
    k = _util.wavenumber(omega, c)
    phi_s = _np.arctan2(n[1], n[0]) + _np.pi
    L = x0.shape[0]

    r = _np.linalg.norm(x0, axis=1)
    phi = _np.arctan2(x0[:, 1], x0[:, 0])
    phi = _np.where(phi < 0, phi + 2 * _np.pi, phi)

    if Nc is None:
        Nc = int(_np.ceil(2 * k * _np.max(r) * alpha / _np.pi))

    epsilon = _np.ones(Nc)  # weights for series expansion
    epsilon[0] = 2

    d = _np.zeros(L, dtype=complex)
    for m in _np.arange(Nc):
        nu = m * _np.pi / alpha
        d = d + 1/epsilon[m] * _np.exp(1j*nu*_np.pi/2) * _np.sin(nu*phi_s) \
            * _np.cos(nu*phi) * nu/r * _jn(nu, k*r)

    d[phi > 0] = -d[phi > 0]

    selection = _util.source_selection_all(len(x0))
    return 4*_np.pi/alpha * d, selection, _secondary_source_line(omega, c)


def plane_2d_edge_dipole_ssd(omega, x0, n=[0, 1, 0], *, alpha=_np.pi*3/2,
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

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.fd.esa.plane_2d_edge_dipole_ssd(
            omega, array.x, npw, alpha=np.pi*3/2)
        plot(d, selection, secondary_source, norm_ref=1)

    """
    x0 = _np.asarray(x0)
    n = _util.normalize_vector(n)
    k = _util.wavenumber(omega, c)
    phi_s = _np.arctan2(n[1], n[0]) + _np.pi
    L = x0.shape[0]

    r = _np.linalg.norm(x0, axis=1)
    phi = _np.arctan2(x0[:, 1], x0[:, 0])
    phi = _np.where(phi < 0, phi + 2 * _np.pi, phi)

    if Nc is None:
        Nc = int(_np.ceil(2 * k * _np.max(r) * alpha / _np.pi))

    epsilon = _np.ones(Nc)  # weights for series expansion
    epsilon[0] = 2

    d = _np.zeros(L, dtype=complex)
    for m in _np.arange(Nc):
        nu = m * _np.pi / alpha
        d = d + 1/epsilon[m] * _np.exp(1j*nu*_np.pi/2) * _np.cos(nu*phi_s) \
            * _np.cos(nu*phi) * _jn(nu, k*r)

    selection = _util.source_selection_all(len(x0))
    return 4*_np.pi/alpha * d, selection, _secondary_source_line(omega, c)


def line_2d_edge(omega, x0, xs, *, alpha=_np.pi*3/2, Nc=None, c=None):
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

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.fd.esa.line_2d_edge(
            omega, array.x, xs, alpha=np.pi*3/2)
        plot(d, selection, secondary_source, norm_ls)

    """
    x0 = _np.asarray(x0)
    k = _util.wavenumber(omega, c)
    phi_s = _np.arctan2(xs[1], xs[0])
    if phi_s < 0:
        phi_s = phi_s + 2 * _np.pi
    r_s = _np.linalg.norm(xs)
    L = x0.shape[0]

    r = _np.linalg.norm(x0, axis=1)
    phi = _np.arctan2(x0[:, 1], x0[:, 0])
    phi = _np.where(phi < 0, phi + 2 * _np.pi, phi)

    if Nc is None:
        Nc = int(_np.ceil(2 * k * _np.max(r) * alpha / _np.pi))

    epsilon = _np.ones(Nc)  # weights for series expansion
    epsilon[0] = 2

    d = _np.zeros(L, dtype=complex)
    idx = (r <= r_s)
    for m in _np.arange(Nc):
        nu = m * _np.pi / alpha
        f = 1/epsilon[m] * _np.sin(nu*phi_s) * _np.cos(nu*phi) * nu/r
        d[idx] = d[idx] + f[idx] * _jn(nu, k*r[idx]) * _hankel2(nu, k*r_s)
        d[~idx] = d[~idx] + f[~idx] * _jn(nu, k*r_s) * _hankel2(nu, k*r[~idx])

    d[phi > 0] = -d[phi > 0]

    selection = _util.source_selection_all(len(x0))
    return -1j*_np.pi/alpha * d, selection, _secondary_source_line(omega, c)


def line_2d_edge_dipole_ssd(omega, x0, xs, *, alpha=_np.pi*3/2, Nc=None,
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

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.fd.esa.line_2d_edge_dipole_ssd(
            omega, array.x, xs, alpha=np.pi*3/2)
        plot(d, selection, secondary_source, norm_ref=1)

    """
    x0 = _np.asarray(x0)
    k = _util.wavenumber(omega, c)
    phi_s = _np.arctan2(xs[1], xs[0])
    if phi_s < 0:
        phi_s = phi_s + 2 * _np.pi
    r_s = _np.linalg.norm(xs)
    L = x0.shape[0]

    r = _np.linalg.norm(x0, axis=1)
    phi = _np.arctan2(x0[:, 1], x0[:, 0])
    phi = _np.where(phi < 0, phi + 2 * _np.pi, phi)

    if Nc is None:
        Nc = int(_np.ceil(2 * k * _np.max(r) * alpha / _np.pi))

    epsilon = _np.ones(Nc)  # weights for series expansion
    epsilon[0] = 2

    d = _np.zeros(L, dtype=complex)
    idx = (r <= r_s)
    for m in _np.arange(Nc):
        nu = m * _np.pi / alpha
        f = 1/epsilon[m] * _np.cos(nu*phi_s) * _np.cos(nu*phi)
        d[idx] = d[idx] + f[idx] * _jn(nu, k*r[idx]) * _hankel2(nu, k*r_s)
        d[~idx] = d[~idx] + f[~idx] * _jn(nu, k*r_s) * _hankel2(nu, k*r[~idx])

    selection = _util.source_selection_all(len(x0))
    return -1j*_np.pi/alpha * d, selection, _secondary_source_line(omega, c)


def point_25d_edge(omega, x0, xs, *, xref=[2, -2, 0], alpha=_np.pi*3/2,
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

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.fd.esa.point_25d_edge(
            omega, array.x, xs, xref=xref, alpha=np.pi*3/2)
        plot(d, selection, secondary_source, norm_ref=1)

    """
    x0 = _np.asarray(x0)
    xs = _np.asarray(xs)
    xref = _np.asarray(xref)

    if _np.isscalar(xref):
        a = _np.linalg.norm(xref) / _np.linalg.norm(xref - xs)
    else:
        a = _np.linalg.norm(xref - x0, axis=1) / _np.linalg.norm(xref - xs)

    d, selection, _ = line_2d_edge(omega, x0, xs, alpha=alpha, Nc=Nc, c=c)
    return 1j*_np.sqrt(a) * d, selection, _secondary_source_point(omega, c)
