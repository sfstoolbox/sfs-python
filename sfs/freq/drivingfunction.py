"""Compute driving functions for various systems.

.. include:: math-definitions.rst

.. plot::
    :context: reset

    import matplotlib.pyplot as plt
    import numpy as np
    import sfs

    plt.rcParams['figure.figsize'] = 6, 6

    xs = -1.5, 1.5, 0
    xs_focused = -0.5, 0.5, 0
    # normal vector for plane wave:
    npw = sfs.util.direction_vector(np.radians(-45))
    # normal vector for focused source:
    ns_focused = sfs.util.direction_vector(np.radians(-45))
    f = 300  # Hz
    omega = 2 * np.pi * f
    R = 1.5  # Radius of circular loudspeaker array

    grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.02)

    array = sfs.array.circular(N=32, R=R)

    def plot(d, selection, secondary_source):
        p = sfs.freq.synthesize(d, selection, array, secondary_source, grid=grid)
        sfs.plot.soundfield(p, grid)
        sfs.plot.loudspeaker_2d(array.x, array.n, selection * array.a, size=0.15)

"""

import numpy as np
from numpy.core.umath_tests import inner1d  # element-wise inner product
from scipy.special import jn, hankel2
from .. import util
from .. import default
from . import source as _source


def wfs_2d_line(omega, x0, n0, xs, c=None):
    r"""Driving function for 2-dimensional WFS for a virtual line source.

    Parameters
    ----------
    omega : float
        Angular frequency of line source.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of normal vectors of secondary sources.
    xs : (3,) array_like
        Position of virtual line source.
    c : float, optional
        Speed of sound.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    .. math::

        D(\x_0,\w) = \frac{\i}{2} \wc
            \frac{\scalarprod{\x-\x_0}{\n_0}}{|\x-\x_\text{s}|}
            \Hankel{2}{1}{\wc|\x-\x_\text{s}|}

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.freq.drivingfunction.wfs_2d_line(
            omega, array.x, array.n, xs)
        plot(d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    xs = util.asarray_1d(xs)
    k = util.wavenumber(omega, c)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    d = -1j/2 * k * inner1d(ds, n0) / r * hankel2(1, k * r)
    selection = util.source_selection_line(n0, x0, xs)
    return d, selection, secondary_source_line(omega, c)


def _wfs_point(omega, x0, n0, xs, c=None):
    r"""Driving function for 2/3-dimensional WFS for a virtual point source.

    Parameters
    ----------
    omega : float
        Angular frequency of point source.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of normal vectors of secondary sources.
    xs : (3,) array_like
        Position of virtual point source.
    c : float, optional
        Speed of sound.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    .. math::

        D(\x_0, \w) = \i\wc \frac{\scalarprod{\x_0-\x_\text{s}}{\n_0}}
            {|\x_0-\x_\text{s}|^{\frac{3}{2}}}
            \e{-\i\wc |\x_0-\x_\text{s}|}

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.freq.drivingfunction.wfs_3d_point(
            omega, array.x, array.n, xs)
        plot(d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    xs = util.asarray_1d(xs)
    k = util.wavenumber(omega, c)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    d = 1j * k * inner1d(ds, n0) / r ** (3 / 2) * np.exp(-1j * k * r)
    selection = util.source_selection_point(n0, x0, xs)
    return d, selection, secondary_source_point(omega, c)


wfs_2d_point = _wfs_point


def wfs_25d_point(omega, x0, n0, xs, xref=[0, 0, 0], c=None, omalias=None):
    r"""Driving function for 2.5-dimensional WFS for a virtual point source.

    Parameters
    ----------
    omega : float
        Angular frequency of point source.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of normal vectors of secondary sources.
    xs : (3,) array_like
        Position of virtual point source.
    xref : (3,) array_like, optional
        Reference point for synthesized sound field.
    c : float, optional
        Speed of sound.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    .. math::

        D(\x_0,\w) = \sqrt{\i\wc |\x_\text{ref}-\x_0|}
            \frac{\scalarprod{\x_0-\x_\text{s}}{\n_0}}
            {|\x_0-\x_\text{s}|^\frac{3}{2}}
            \e{-\i\wc |\x_0-\x_\text{s}|}

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.freq.drivingfunction.wfs_25d_point(
            omega, array.x, array.n, xs)
        plot(d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    xs = util.asarray_1d(xs)
    xref = util.asarray_1d(xref)
    k = util.wavenumber(omega, c)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    d = (
        wfs_25d_preeq(omega, omalias, c) *
        np.sqrt(np.linalg.norm(xref - x0)) * inner1d(ds, n0) /
        r ** (3 / 2) * np.exp(-1j * k * r))
    selection = util.source_selection_point(n0, x0, xs)
    return d, selection, secondary_source_point(omega, c)


wfs_3d_point = _wfs_point


def _wfs_plane(omega, x0, n0, n=[0, 1, 0], c=None):
    r"""Driving function for 2/3-dimensional WFS for a virtual plane wave.

    Parameters
    ----------
    omega : float
        Angular frequency of plane wave.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of normal vectors of secondary sources.
    n : (3,) array_like, optional
        Normal vector (traveling direction) of plane wave.
    c : float, optional
        Speed of sound.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    Eq.(17) from :cite:`Spors2008`:

    .. math::

        D(\x_0,\w) = \i\wc \scalarprod{\n}{\n_0}
            \e{-\i\wc\scalarprod{\n}{\x_0}}

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.freq.drivingfunction.wfs_3d_plane(
            omega, array.x, array.n, npw)
        plot(d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    n = util.normalize_vector(n)
    k = util.wavenumber(omega, c)
    d = 2j * k * np.inner(n, n0) * np.exp(-1j * k * np.inner(n, x0))
    selection = util.source_selection_plane(n0, n)
    return d, selection, secondary_source_point(omega, c)


wfs_2d_plane = _wfs_plane


def wfs_25d_plane(omega, x0, n0, n=[0, 1, 0], xref=[0, 0, 0], c=None,
                  omalias=None):
    r"""Driving function for 2.5-dimensional WFS for a virtual plane wave.

    Parameters
    ----------
    omega : float
        Angular frequency of plane wave.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of normal vectors of secondary sources.
    n : (3,) array_like, optional
        Normal vector (traveling direction) of plane wave.
    xref : (3,) array_like, optional
        Reference point for synthesized sound field.
    c : float, optional
        Speed of sound.
    omalias: float, optional
        Angular frequency where spatial aliasing becomes prominent.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    .. math::

        D_\text{2.5D}(\x_0,\w) = \sqrt{\i\wc |\x_\text{ref}-\x_0|}
            \scalarprod{\n}{\n_0}
            \e{-\i\wc \scalarprod{\n}{\x_0}}

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.freq.drivingfunction.wfs_25d_plane(
            omega, array.x, array.n, npw)
        plot(d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    n = util.normalize_vector(n)
    xref = util.asarray_1d(xref)
    k = util.wavenumber(omega, c)
    d = (
        wfs_25d_preeq(omega, omalias, c) *
        np.sqrt(8*np.pi * np.linalg.norm(xref - x0, axis=-1)) *
        np.inner(n, n0) * np.exp(-1j * k * np.inner(n, x0)))
    selection = util.source_selection_plane(n0, n)
    return d, selection, secondary_source_point(omega, c)


wfs_3d_plane = _wfs_plane


def _wfs_focused(omega, x0, n0, xs, ns, c=None):
    r"""Driving function for 2/3-dimensional WFS for a focused source.

    Parameters
    ----------
    omega : float
        Angular frequency of focused source.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of normal vectors of secondary sources.
    xs : (3,) array_like
        Position of focused source.
    ns :  (3,) array_like
        Direction of focused source.
    c : float, optional
        Speed of sound.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    .. math::

        D(\x_0,\w) = \i\wc \frac{\scalarprod{\x_0-\x_\text{s}}{\n_0}}
            {|\x_0-\x_\text{s}|^\frac{3}{2}}
            \e{\i\wc |\x_0-\x_\text{s}|}

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.freq.drivingfunction.wfs_3d_focused(
            omega, array.x, array.n, xs_focused, ns_focused)
        plot(d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    xs = util.asarray_1d(xs)
    k = util.wavenumber(omega, c)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    d = 1j * k * inner1d(ds, n0) / r ** (3 / 2) * np.exp(1j * k * r)
    selection = util.source_selection_focused(ns, x0, xs)
    return d, selection, secondary_source_point(omega, c)


wfs_2d_focused = _wfs_focused


def wfs_25d_focused(omega, x0, n0, xs, ns, xref=[0, 0, 0], c=None,
                    omalias=None):
    r"""Driving function for 2.5-dimensional WFS for a focused source.

    Parameters
    ----------
    omega : float
        Angular frequency of focused source.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of normal vectors of secondary sources.
    xs : (3,) array_like
        Position of focused source.
    ns :  (3,) array_like
        Direction of focused source.
    xref : (3,) array_like, optional
        Reference point for synthesized sound field.
    c : float, optional
        Speed of sound.
    omalias: float, optional
        Angular frequency where spatial aliasing becomes prominent.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    .. math::

        D(\x_0,\w) = \sqrt{\i\wc |\x_\text{ref}-\x_0|}
            \frac{\scalarprod{\x_0-\x_\text{s}}{\n_0}}
            {|\x_0-\x_\text{s}|^\frac{3}{2}}
            \e{\i\wc |\x_0-\x_\text{s}|}

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.freq.drivingfunction.wfs_25d_focused(
            omega, array.x, array.n, xs_focused, ns_focused)
        plot(d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    xs = util.asarray_1d(xs)
    xref = util.asarray_1d(xref)
    k = util.wavenumber(omega, c)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    d = (
        wfs_25d_preeq(omega, omalias, c) *
        np.sqrt(np.linalg.norm(xref - x0)) * inner1d(ds, n0) /
        r ** (3 / 2) * np.exp(1j * k * r))
    selection = util.source_selection_focused(ns, x0, xs)
    return d, selection, secondary_source_point(omega, c)


wfs_3d_focused = _wfs_focused


def wfs_25d_preeq(omega, omalias, c):
    r"""Pre-equalization filter for 2.5-dimensional WFS.

    Parameters
    ----------
    omega : float
        Angular frequency.
    omalias: float
        Angular frequency where spatial aliasing becomes prominent.
    c : float
        Speed of sound.

    Returns
    -------
    float
        Complex weight for given angular frequency.

    Notes
    -----
    .. math::

        H(\w) = \begin{cases}
            \sqrt{\i \wc} & \text{for } \w \leq \w_\text{alias} \\
            \sqrt{\i \frac{\w_\text{alias}}{c}} & \text{for } \w > \w_\text{alias}
            \end{cases}

    """
    if omalias is None:
        return np.sqrt(1j * util.wavenumber(omega, c))
    else:
        if omega <= omalias:
            return np.sqrt(1j * util.wavenumber(omega, c))
        else:
            return np.sqrt(1j * util.wavenumber(omalias, c))


def delay_3d_plane(omega, x0, n0, n=[0, 1, 0], c=None):
    r"""Delay-only driving function for a virtual plane wave.

    Parameters
    ----------
    omega : float
        Angular frequency of plane wave.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of normal vectors of secondary sources.
    n : (3,) array_like, optional
        Normal vector (traveling direction) of plane wave.
    c : float, optional
        Speed of sound.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    .. math::

        D(\x_0,\w) = \e{-\i\wc\scalarprod{\n}{\x_0}}

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.mono.drivingfunction.delay_3d_plane(
            omega, array.x, array.n, npw)
        plot(d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n = util.normalize_vector(n)
    k = util.wavenumber(omega, c)
    d = np.exp(-1j * k * np.inner(n, x0))
    selection = util.source_selection_plane(n0, n)
    return d, selection, secondary_source_point(omega, c)


def nfchoa_2d_plane(omega, x0, r0, n=[0, 1, 0], max_order=None, c=None):
    r"""Driving function for 2-dimensional NFC-HOA for a virtual plane wave.

    Parameters
    ----------
    omega : float
        Angular frequency of plane wave.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    r0 : float
        Radius of circular secondary source distribution.
    n : (3,) array_like, optional
        Normal vector (traveling direction) of plane wave.
    max_order : float, optional
        Maximum order of circular harmonics used for the calculation.
    c : float, optional
        Speed of sound.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing only ``True`` indicating that
        all secondary source are "active" for NFC-HOA.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    .. math::

        D(\phi_0, \omega) =
        -\frac{2\i}{\pi r_0}
        \sum_{m=-M}^M
        \frac{\i^{-m}}{\Hankel{2}{m}{\wc r_0}}
        \e{\i m (\phi_0 - \phi_\text{pw})}

    See http://sfstoolbox.org/#equation-D.nfchoa.pw.2D.

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.freq.drivingfunction.nfchoa_2d_plane(
            omega, array.x, R, npw)
        plot(d, selection, secondary_source)

    """
    if max_order is None:
        max_order = util.max_order_circular_harmonics(len(x0))

    x0 = util.asarray_of_rows(x0)
    k = util.wavenumber(omega, c)
    n = util.normalize_vector(n)
    phi, _, r = util.cart2sph(*n)
    phi0 = util.cart2sph(*x0.T)[0]
    d = 0
    for m in range(-max_order, max_order + 1):
        d += 1j**-m / hankel2(m, k * r0) * np.exp(1j * m * (phi0 - phi))
    selection = util.source_selection_all(len(x0))
    return -2j / (np.pi*r0) * d, selection, secondary_source_point(omega, c)


def nfchoa_25d_point(omega, x0, r0, xs, max_order=None, c=None):
    r"""Driving function for 2.5-dimensional NFC-HOA for a virtual point source.

    Parameters
    ----------
    omega : float
        Angular frequency of point source.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    r0 : float
        Radius of circular secondary source distribution.
    xs : (3,) array_like
        Position of point source.
    max_order : float, optional
        Maximum order of circular harmonics used for the calculation.
    c : float, optional
        Speed of sound.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing only ``True`` indicating that
        all secondary source are "active" for NFC-HOA.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    .. math::

        D(\phi_0, \omega) =
        \frac{1}{2 \pi r_0}
        \sum_{m=-M}^M
        \frac{\hankel{2}{|m|}{\wc r}}{\hankel{2}{|m|}{\wc r_0}}
        \e{\i m (\phi_0 - \phi)}

    See http://sfstoolbox.org/#equation-D.nfchoa.ps.2.5D.

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.freq.drivingfunction.nfchoa_25d_point(
            omega, array.x, R, xs)
        plot(d, selection, secondary_source)

    """
    if max_order is None:
        max_order = util.max_order_circular_harmonics(len(x0))

    x0 = util.asarray_of_rows(x0)
    k = util.wavenumber(omega, c)
    xs = util.asarray_1d(xs)
    phi, _, r = util.cart2sph(*xs)
    phi0 = util.cart2sph(*x0.T)[0]
    hr = util.spherical_hn2(range(0, max_order + 1), k * r)
    hr0 = util.spherical_hn2(range(0, max_order + 1), k * r0)
    d = 0
    for m in range(-max_order, max_order + 1):
        d += hr[abs(m)] / hr0[abs(m)] * np.exp(1j * m * (phi0 - phi))
    selection = util.source_selection_all(len(x0))
    return d / (2 * np.pi * r0), selection, secondary_source_point(omega, c)


def nfchoa_25d_plane(omega, x0, r0, n=[0, 1, 0], max_order=None, c=None):
    r"""Driving function for 2.5-dimensional NFC-HOA for a virtual plane wave.

    Parameters
    ----------
    omega : float
        Angular frequency of point source.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    r0 : float
        Radius of circular secondary source distribution.
    n : (3,) array_like, optional
        Normal vector (traveling direction) of plane wave.
    max_order : float, optional
        Maximum order of circular harmonics used for the calculation.
    c : float, optional
        Speed of sound.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing only ``True`` indicating that
        all secondary source are "active" for NFC-HOA.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    .. math::

        D(\phi_0, \omega) =
        \frac{2\i}{r_0}
        \sum_{m=-M}^M
        \frac{\i^{-|m|}}{\wc \hankel{2}{|m|}{\wc r_0}}
        \e{\i m (\phi_0 - \phi_\text{pw})}

    See http://sfstoolbox.org/#equation-D.nfchoa.pw.2.5D.

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.freq.drivingfunction.nfchoa_25d_plane(
            omega, array.x, R, npw)
        plot(d, selection, secondary_source)

    """
    if max_order is None:
        max_order = util.max_order_circular_harmonics(len(x0))

    x0 = util.asarray_of_rows(x0)
    k = util.wavenumber(omega, c)
    n = util.normalize_vector(n)
    phi, _, r = util.cart2sph(*n)
    phi0 = util.cart2sph(*x0.T)[0]
    d = 0
    hn2 = util.spherical_hn2(range(0, max_order + 1), k * r0)
    for m in range(-max_order, max_order + 1):
        d += (-1j)**abs(m) / (k * hn2[abs(m)]) * np.exp(1j * m * (phi0 - phi))
    selection = util.source_selection_all(len(x0))
    return 2*1j / r0 * d, selection, secondary_source_point(omega, c)


def sdm_2d_line(omega, x0, n0, xs, c=None):
    r"""Driving function for 2-dimensional SDM for a virtual line source.

    Parameters
    ----------
    omega : float
        Angular frequency of line source.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of normal vectors of secondary sources.
    xs : (3,) array_like
        Position of line source.
    c : float, optional
        Speed of sound.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    The secondary sources have to be located on the x-axis (y0=0).
    Derived from :cite:`Spors2009`, Eq.(9), Eq.(4).

    Examples
    --------
    .. plot::
        :context: close-figs

        array = sfs.array.linear(32, 0.2, orientation=[0, -1, 0])
        d, selection, secondary_source = sfs.mono.drivingfunction.sdm_2d_line(
            omega, array.x, array.n, xs)
        plot(d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    xs = util.asarray_1d(xs)
    k = util.wavenumber(omega, c)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    d = - 1j/2 * k * xs[1] / r * hankel2(1, k * r)
    selection = util.source_selection_all(len(x0))
    return d, selection, secondary_source_line(omega, c)


def sdm_2d_plane(omega, x0, n0, n=[0, 1, 0], c=None):
    r"""Driving function for 2-dimensional SDM for a virtual plane wave.

    Parameters
    ----------
    omega : float
        Angular frequency of plane wave.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of normal vectors of secondary sources.
    n: (3,) array_like, optional
        Normal vector (traveling direction) of plane wave.
    c : float, optional
        Speed of sound.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    The secondary sources have to be located on the x-axis (y0=0).
    Derived from :cite:`Ahrens2012`, Eq.(3.73), Eq.(C.5), Eq.(C.11):

    .. math::

        D(\x_0,k) = k_\text{pw,y} \e{-\i k_\text{pw,x} x}

    Examples
    --------
    .. plot::
        :context: close-figs

        array = sfs.array.linear(32, 0.2, orientation=[0, -1, 0])
        d, selection, secondary_source = sfs.mono.drivingfunction.sdm_2d_plane(
            omega, array.x, array.n, npw)
        plot(d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    n = util.normalize_vector(n)
    k = util.wavenumber(omega, c)
    d = k * n[1] * np.exp(-1j * k * n[0] * x0[:, 0])
    selection = util.source_selection_all(len(x0))
    return d, selection, secondary_source_line(omega, c)


def sdm_25d_plane(omega, x0, n0, n=[0, 1, 0], xref=[0, 0, 0], c=None):
    r"""Driving function for 2.5-dimensional SDM for a virtual plane wave.

    Parameters
    ----------
    omega : float
        Angular frequency of plane wave.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of normal vectors of secondary sources.
    n: (3,) array_like, optional
        Normal vector (traveling direction) of plane wave.
    xref : (3,) array_like, optional
        Reference point for synthesized sound field.
    c : float, optional
        Speed of sound.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    The secondary sources have to be located on the x-axis (y0=0).
    Eq.(3.79) from :cite:`Ahrens2012`.

    Examples
    --------
    .. plot::
        :context: close-figs

        array = sfs.array.linear(32, 0.2, orientation=[0, -1, 0])
        d, selection, secondary_source = (
            sfs.mono.drivingfunction.sdm_25d_plane(
                omega, array.x, array.n, npw, [0, -1, 0]))
        plot(d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    n = util.normalize_vector(n)
    xref = util.asarray_1d(xref)
    k = util.wavenumber(omega, c)
    d = 4j * np.exp(-1j*k*n[1]*xref[1]) / hankel2(0, k*n[1]*xref[1]) * \
        np.exp(-1j*k*n[0]*x0[:, 0])
    selection = util.source_selection_all(len(x0))
    return d, selection, secondary_source_point(omega, c)


def sdm_25d_point(omega, x0, n0, xs, xref=[0, 0, 0], c=None):
    r"""Driving function for 2.5-dimensional SDM for a virtual point source.

    Parameters
    ----------
    omega : float
        Angular frequency of point source.
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of normal vectors of secondary sources.
    xs: (3,) array_like
        Position of virtual point source.
    xref : (3,) array_like, optional
        Reference point for synthesized sound field.
    c : float, optional
        Speed of sound.

    Returns
    -------
    d : (N,) numpy.ndarray
        Complex weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.mono.synthesize()`.

    Notes
    -----
    The secondary sources have to be located on the x-axis (y0=0).
    Driving function from :cite:`Spors2010`, Eq.(24).

    Examples
    --------
    .. plot::
        :context: close-figs

        array = sfs.array.linear(32, 0.2, orientation=[0, -1, 0])
        d, selection, secondary_source = (
            sfs.mono.drivingfunction.sdm_25d_point(
                omega, array.x, array.n, xs, [0, -1, 0]))
        plot(d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    xs = util.asarray_1d(xs)
    xref = util.asarray_1d(xref)
    k = util.wavenumber(omega, c)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    d = 1/2 * 1j * k * np.sqrt(xref[1] / (xref[1] - xs[1])) * \
        xs[1] / r * hankel2(1, k * r)
    selection = util.source_selection_all(len(x0))
    return d, selection, secondary_source_point(omega, c)


def esa_edge_2d_plane(omega, x0, n=[0, 1, 0], alpha=3/2*np.pi, Nc=None,
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
        single secondary source.  See `sfs.mono.synthesize()`.

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


def esa_edge_dipole_2d_plane(omega, x0, n=[0, 1, 0], alpha=3/2*np.pi, Nc=None,
                             c=None):
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
        single secondary source.  See `sfs.mono.synthesize()`.

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


def esa_edge_2d_line(omega, x0, xs, alpha=3/2*np.pi, Nc=None, c=None):
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
        single secondary source.  See `sfs.mono.synthesize()`.

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


def esa_edge_dipole_2d_line(omega, x0, xs, alpha=3/2*np.pi, Nc=None, c=None):
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
        single secondary source.  See `sfs.mono.synthesize()`.

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


def esa_edge_25d_point(omega, x0, xs, xref=[2, -2, 0], alpha=3/2*np.pi,
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
        single secondary source.  See `sfs.mono.synthesize()`.

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

    d, selection, _ = esa_edge_2d_line(omega, x0, xs, alpha=alpha, Nc=Nc, c=c)
    return 1j*np.sqrt(a) * d, selection, secondary_source_point(omega, c)


def secondary_source_point(omega, c):
    """Create a point source for use in `sfs.freq.synthesize()`."""

    def secondary_source(position, _, grid):
        return _source.point(omega, position, grid, c)

    return secondary_source


def secondary_source_line(omega, c):
    """Create a line source for use in `sfs.freq.synthesize()`."""

    def secondary_source(position, _, grid):
        return _source.line(omega, position, grid, c)

    return secondary_source
