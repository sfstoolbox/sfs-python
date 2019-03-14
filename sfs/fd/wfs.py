"""Compute WFS driving functions.

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
        p = sfs.fd.synthesize(d, selection, array, secondary_source, grid=grid)
        sfs.plot2d.amplitude(p, grid)
        sfs.plot2d.loudspeakers(array.x, array.n, selection * array.a, size=0.15)

"""

import numpy as np
from numpy.core.umath_tests import inner1d  # element-wise inner product
from scipy.special import hankel2
from .. import util
from . import secondary_source_line, secondary_source_point


def line_2d(omega, x0, n0, xs, *, c=None):
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
        single secondary source.  See `sfs.fd.synthesize()`.

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

        d, selection, secondary_source = sfs.fd.wfs.line_2d(
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


def _point(omega, x0, n0, xs, *, c=None):
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
        single secondary source.  See `sfs.fd.synthesize()`.

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

        d, selection, secondary_source = sfs.fd.wfs.point_3d(
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


point_2d = _point


def point_25d(omega, x0, n0, xs, xref=[0, 0, 0], c=None, omalias=None):
    r"""Driving function for 2.5-dimensional WFS of a virtual point source.

    .. versionchanged:: 0.5
        see notes, old handling of `point_25d()` is now `point_25d_legacy()`

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
        Reference point xref or contour xref(x0) for amplitude correct
        synthesis.
    c : float, optional
        Speed of sound in m/s.
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
        single secondary source.  See `sfs.fd.synthesize()`.

    Notes
    -----
    `point_25d()` derives 2.5D WFS from the 3D
    Neumann-Rayleigh integral (i.e. the TU Delft approach).
    The eq. (3.10), (3.11) in :cite:`Start1997`, equivalent to
    Eq. (2.137) in :cite:`Schultz2016`

    .. math::

        D(\x_0,\w) = \sqrt{8 \pi \, \i\wc}
            \sqrt{\frac{|\x_\text{ref}-\x_0| \cdot
            |\x_0-\x_\text{s}|}{|\x_\text{ref}-\x_0| + |\x_0-\x_\text{s}|}}
            \scalarprod{\frac{\x_0-\x_\text{s}}{|\x_0-\x_\text{s}|}}{\n_0}
            \frac{\e{-\i\wc |\x_0-\x_\text{s}|}}{4\pi\,|\x_0-\x_\text{s}|}

    is implemented.
    The theoretical link of `point_25d()` and `point_25d_legacy()` was
    introduced as *unified WFS framework* in :cite:`Firtha2017`.

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.fd.wfs.point_25d(
            omega, array.x, array.n, xs)
        normalize_gain = 4 * np.pi * np.linalg.norm(xs)
        plot(normalize_gain * d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    xs = util.asarray_1d(xs)
    xref = util.asarray_1d(xref)
    k = util.wavenumber(omega, c)

    ds = x0 - xs
    dr = xref - x0
    s = np.linalg.norm(ds, axis=1)
    r = np.linalg.norm(dr, axis=1)

    d = (
        preeq_25d(omega, omalias, c) *
        np.sqrt(8 * np.pi) *
        np.sqrt((r * s) / (r + s)) *
        inner1d(n0, ds) / s *
        np.exp(-1j * k * s) / (4 * np.pi * s))
    selection = util.source_selection_point(n0, x0, xs)
    return d, selection, secondary_source_point(omega, c)


point_3d = _point


def point_25d_legacy(omega, x0, n0, xs, xref=[0, 0, 0], c=None, omalias=None):
    r"""Driving function for 2.5-dimensional WFS for a virtual point source.

    .. versionadded:: 0.5
        `point_25d()` was renamed to `point_25d_legacy()` (and a new
        function with the name `point_25d()` was introduced). See notes for
        further details.

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
        single secondary source.  See `sfs.fd.synthesize()`.

    Notes
    -----
    `point_25d_legacy()` derives 2.5D WFS from the 2D
    Neumann-Rayleigh integral (i.e. the approach by Rabenstein & Spors), cf.
    :cite:`Spors2008`.

    .. math::

        D(\x_0,\w) = \sqrt{\i\wc |\x_\text{ref}-\x_0|}
            \frac{\scalarprod{\x_0-\x_\text{s}}{\n_0}}
            {|\x_0-\x_\text{s}|^\frac{3}{2}}
            \e{-\i\wc |\x_0-\x_\text{s}|}

    The theoretical link of `point_25d()` and `point_25d_legacy()` was
    introduced as *unified WFS framework* in :cite:`Firtha2017`.
    Also cf. Eq. (2.145)-(2.147) :cite:`Schultz2016`.

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.fd.wfs.point_25d_legacy(
            omega, array.x, array.n, xs)
        normalize_gain = np.linalg.norm(xs)
        plot(normalize_gain * d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    xs = util.asarray_1d(xs)
    xref = util.asarray_1d(xref)
    k = util.wavenumber(omega, c)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    d = (
        preeq_25d(omega, omalias, c) *
        np.sqrt(np.linalg.norm(xref - x0)) * inner1d(ds, n0) /
        r ** (3 / 2) * np.exp(-1j * k * r))
    selection = util.source_selection_point(n0, x0, xs)
    return d, selection, secondary_source_point(omega, c)


def _plane(omega, x0, n0, n=[0, 1, 0], *, c=None):
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
        single secondary source.  See `sfs.fd.synthesize()`.

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

        d, selection, secondary_source = sfs.fd.wfs.plane_3d(
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


plane_2d = _plane


def plane_25d(omega, x0, n0, n=[0, 1, 0], *, xref=[0, 0, 0], c=None,
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
        single secondary source.  See `sfs.fd.synthesize()`.

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

        d, selection, secondary_source = sfs.fd.wfs.plane_25d(
            omega, array.x, array.n, npw)
        plot(d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    n = util.normalize_vector(n)
    xref = util.asarray_1d(xref)
    k = util.wavenumber(omega, c)
    d = (
        preeq_25d(omega, omalias, c) *
        np.sqrt(8*np.pi * np.linalg.norm(xref - x0, axis=-1)) *
        np.inner(n, n0) * np.exp(-1j * k * np.inner(n, x0)))
    selection = util.source_selection_plane(n0, n)
    return d, selection, secondary_source_point(omega, c)


plane_3d = _plane


def _focused(omega, x0, n0, xs, ns, *, c=None):
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
        single secondary source.  See `sfs.fd.synthesize()`.

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

        d, selection, secondary_source = sfs.fd.wfs.focused_3d(
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


focused_2d = _focused


def focused_25d(omega, x0, n0, xs, ns, *, xref=[0, 0, 0], c=None,
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
        single secondary source.  See `sfs.fd.synthesize()`.

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

        d, selection, secondary_source = sfs.fd.wfs.focused_25d(
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
        preeq_25d(omega, omalias, c) *
        np.sqrt(np.linalg.norm(xref - x0)) * inner1d(ds, n0) /
        r ** (3 / 2) * np.exp(1j * k * r))
    selection = util.source_selection_focused(ns, x0, xs)
    return d, selection, secondary_source_point(omega, c)


focused_3d = _focused


def preeq_25d(omega, omalias, c):
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
            \sqrt{\i \frac{\w_\text{alias}}{c}} &
            \text{for } \w > \w_\text{alias}
            \end{cases}

    """
    if omalias is None:
        return np.sqrt(1j * util.wavenumber(omega, c))
    else:
        if omega <= omalias:
            return np.sqrt(1j * util.wavenumber(omega, c))
        else:
            return np.sqrt(1j * util.wavenumber(omalias, c))


def plane_3d_delay(omega, x0, n0, n=[0, 1, 0], *, c=None):
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
        single secondary source.  See `sfs.fd.synthesize()`.

    Notes
    -----
    .. math::

        D(\x_0,\w) = \e{-\i\wc\scalarprod{\n}{\x_0}}

    Examples
    --------
    .. plot::
        :context: close-figs

        d, selection, secondary_source = sfs.fd.wfs.plane_3d_delay(
            omega, array.x, array.n, npw)
        plot(d, selection, secondary_source)

    """
    x0 = util.asarray_of_rows(x0)
    n = util.normalize_vector(n)
    k = util.wavenumber(omega, c)
    d = np.exp(-1j * k * np.inner(n, x0))
    selection = util.source_selection_plane(n0, n)
    return d, selection, secondary_source_point(omega, c)


def soundfigure_3d(omega, x0, n0, figure, npw=[0, 0, 1], *, c=None):
    """Compute driving function for a 2D sound figure.

    Based on
    [Helwani et al., The Synthesis of Sound Figures, MSSP, 2013]

    """
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    k = util.wavenumber(omega, c)
    nx, ny = figure.shape

    # 2D spatial DFT of image
    figure = np.fft.fftshift(figure, axes=(0, 1))  # sign of spatial DFT
    figure = np.fft.fft2(figure)
    # wavenumbers
    kx = np.fft.fftfreq(nx, 1./nx)
    ky = np.fft.fftfreq(ny, 1./ny)
    # shift spectrum due to desired plane wave
    figure = np.roll(figure, int(k*npw[0]), axis=0)
    figure = np.roll(figure, int(k*npw[1]), axis=1)
    # search and iterate over propagating plane wave components
    kxx, kyy = np.meshgrid(kx, ky, sparse=True)
    rho = np.sqrt((kxx) ** 2 + (kyy) ** 2)
    d = 0
    for n in range(nx):
        for m in range(ny):
            if(rho[n, m] < k):
                # dispertion relation
                kz = np.sqrt(k**2 - rho[n, m]**2)
                # normal vector of plane wave
                npw = 1/k * np.asarray([kx[n], ky[m], kz])
                npw = npw / np.linalg.norm(npw)
                # driving function of plane wave with positive kz
                d_component, selection, secondary_source = plane_3d(
                    omega, x0, n0, npw, c=c)
                d += selection * figure[n, m] * d_component

    return d, util.source_selection_all(len(d)), secondary_source
