"""Compute NFC-HOA driving functions.

.. include:: math-definitions.rst

.. plot::
    :context: reset

    import matplotlib.pyplot as plt
    import numpy as np
    import sfs
    from scipy.signal import unit_impulse

    # Plane wave
    npw = sfs.util.direction_vector(np.radians(-45))

    # Point source
    xs = -1.5, 1.5, 0
    rs = np.linalg.norm(xs)  # distance from origin
    ts = rs / sfs.default.c  # time-of-arrival at origin

    # Impulsive excitation
    fs = 44100
    signal = unit_impulse(512), fs

    # Circular loudspeaker array
    N = 32  # number of loudspeakers
    R = 1.5  # radius
    array = sfs.array.circular(N, R)

    grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.02)

    def plot(d, selection, secondary_source, t=0):
        p = sfs.time.synthesize(d, selection, array, secondary_source, grid=grid,
                                observation_time=t)
        sfs.plot.level(p, grid)
        sfs.plot.loudspeaker_2d(array.x, array.n, selection * array.a, size=0.15)

"""
import numpy as np
from scipy.signal import besselap, sosfilt, zpk2sos
from scipy.special import eval_legendre as legendre
from .. import default
from .. import util
from . import secondary_source_point


def matchedz_zpk(s_zeros, s_poles, s_gain, fs):
    """Matched-z transform of poles and zeros.

    Parameters
    ----------
    s_zeros : array_like
        Zeros in the Laplace domain.
    s_poles : array_like
        Poles in the Laplace domain.
    s_gain : float
        System gain in the Laplace domain.
    fs : int
        Sampling frequency in Hertz.

    Returns
    -------
    z_zeros : numpy.ndarray
        Zeros in the z-domain.
    z_poles : numpy.ndarray
        Poles in the z-domain.
    z_gain : float
        System gain in the z-domain.

    See Also
    --------
    :func:`scipy.signal.bilinear_zpk`

    """
    z_zeros = np.exp(s_zeros / fs)
    z_poles = np.exp(s_poles / fs)
    omega = 1j * np.pi * fs
    s_gain *= np.prod((omega - s_zeros) / (omega - s_poles)
                      * (-1 - z_poles) / (-1 - z_zeros))
    return z_zeros, z_poles, np.real(s_gain)


def plane_25d(x0, r0, npw, fs, max_order=None, c=None, s2z=matchedz_zpk):
    r"""Virtual plane wave by 2.5-dimensional NFC-HOA.

    .. math::

        D(\phi_0, s) =
        2\e{\frac{s}{c}r_0}
        \sum_{m=-M}^{M}
        (-1)^m
        \Big(\frac{s}{s-\frac{c}{r_0}\sigma_0}\Big)^\mu
        \prod_{l=1}^{\nu}
        \frac{s^2}{(s-\frac{c}{r_0}\sigma_l)^2+(\frac{c}{r_0}\omega_l)^2}
        \e{\i m(\phi_0 - \phi_\text{pw})}

    The driving function is represented in the Laplace domain,
    from which the recursive filters are designed.
    :math:`\sigma_l + \i\omega_l` denotes the complex roots of
    the reverse Bessel polynomial.
    The number of second-order sections is
    :math:`\nu = \big\lfloor\tfrac{|m|}{2}\big\rfloor`,
    whereas the number of first-order section :math:`\mu` is either 0 or 1
    for even and odd :math:`|m|`, respectively.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    r0 : float
        Radius of the circular secondary source distribution.
    npw : (3,) array_like
        Unit vector (propagation direction) of plane wave.
    fs : int
        Sampling frequency in Hertz.
    max_order : int, optional
        Ambisonics order.
    c : float, optional
        Speed of sound in m/s.
    s2z : callable, optional
        Function transforming s-domain poles and zeros into z-domain,
        e.g. :func:`matchedz_zpk`, :func:`scipy.signal.bilinear_zpk`.

    Returns
    -------
    delay : float
        Overall delay in seconds.
    weight : float
        Overall weight.
    sos : list of numpy.ndarray
        Second-order section filters :func:`scipy.signal.sosfilt`.
    phaseshift : (N,) numpy.ndarray
        Phase shift in radians.
    selection : (N,) numpy.ndarray
        Boolean array containing only ``True`` indicating that
        all secondary source are "active" for NFC-HOA.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.time.synthesize()`.

    Examples
    --------
    .. plot::
        :context: close-figs

        delay, weight, sos, phaseshift, selection, secondary_source = \
            sfs.time.nfchoa.plane_25d(array.x, R, npw, fs)
        d = sfs.time.nfchoa.driving_signals_25d(
                delay, weight, sos, phaseshift, signal)
        plot(d, selection, secondary_source)

    """
    if max_order is None:
        max_order = util.max_order_circular_harmonics(len(x0))
    if c is None:
        c = default.c

    x0 = util.asarray_of_rows(x0)
    npw = util.asarray_1d(npw)
    phi0, _, _ = util.cart2sph(*x0.T)
    phipw, _, _ = util.cart2sph(*npw)
    phaseshift = phi0 - phipw + np.pi

    delay = -r0 / c
    weight = 2
    sos = []
    for m in range(max_order + 1):
        _, p, _ = besselap(m, norm='delay')
        s_zeros = np.zeros(m)
        s_poles = c / r0 * p
        s_gain = 1
        z_zeros, z_poles, z_gain = s2z(s_zeros, s_poles, s_gain, fs)
        sos.append(zpk2sos(z_zeros, z_poles, z_gain, pairing='nearest'))
    selection = util.source_selection_all(len(x0))
    return delay, weight, sos, phaseshift, selection, secondary_source_point(c)


def point_25d(x0, r0, xs, fs, max_order=None, c=None, s2z=matchedz_zpk):
    r"""Virtual Point source by 2.5-dimensional NFC-HOA.

    .. math::

        D(\phi_0, s) =
        \frac{1}{2\pi r_\text{s}}
        \e{\frac{s}{c}(r_0-r_\text{s})}
        \sum_{m=-M}^{M}
        \Big(\frac{s-\frac{c}{r_\text{s}}\sigma_0}{s-\frac{c}{r_0}\sigma_0}\Big)^\mu
        \prod_{l=1}^{\nu}
        \frac{(s-\frac{c}{r_\text{s}}\sigma_l)^2-(\frac{c}{r_\text{s}}\omega_l)^2}
        {(s-\frac{c}{r_0}\sigma_l)^2+(\frac{c}{r_0}\omega_l)^2}
        \e{\i m(\phi_0 - \phi_\text{s})}

    The driving function is represented in the Laplace domain,
    from which the recursive filters are designed.
    :math:`\sigma_l + \i\omega_l` denotes the complex roots of
    the reverse Bessel polynomial.
    The number of second-order sections is
    :math:`\nu = \big\lfloor\tfrac{|m|}{2}\big\rfloor`,
    whereas the number of first-order section :math:`\mu` is either 0 or 1
    for even and odd :math:`|m|`, respectively.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    r0 : float
        Radius of the circular secondary source distribution.
    xs : (3,) array_like
        Virtual source position.
    fs : int
        Sampling frequency in Hertz.
    max_order : int, optional
        Ambisonics order.
    c : float, optional
        Speed of sound in m/s.
    s2z : callable, optional
        Function transforming s-domain poles and zeros into z-domain,
        e.g. :func:`matchedz_zpk`, :func:`scipy.signal.bilinear_zpk`.

    Returns
    -------
    delay : float
        Overall delay in seconds.
    weight : float
        Overall weight.
    sos : list of numpy.ndarray
        Second-order section filters :func:`scipy.signal.sosfilt`.
    phaseshift : (N,) numpy.ndarray
        Phase shift in radians.
    selection : (N,) numpy.ndarray
        Boolean array containing only ``True`` indicating that
        all secondary source are "active" for NFC-HOA.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.time.synthesize()`.

    Examples
    --------
    .. plot::
        :context: close-figs

        delay, weight, sos, phaseshift, selection, secondary_source = \
            sfs.time.nfchoa.point_25d(array.x, R, xs, fs)
        d = sfs.time.nfchoa.driving_signals_25d(
                delay, weight, sos, phaseshift, signal)
        plot(d, selection, secondary_source, t=ts)

    """
    if max_order is None:
        max_order = util.max_order_circular_harmonics(len(x0))
    if c is None:
        c = default.c

    x0 = util.asarray_of_rows(x0)
    xs = util.asarray_1d(xs)
    phi0, _, _ = util.cart2sph(*x0.T)
    phis, _, rs = util.cart2sph(*xs)
    phaseshift = phi0 - phis

    delay = (rs - r0) / c
    weight = 1 / 2 / np.pi / rs
    sos = []
    for m in range(max_order + 1):
        _, p, _ = besselap(m, norm='delay')
        s_zeros = c / rs * p
        s_poles = c / r0 * p
        s_gain = 1
        z_zeros, z_poles, z_gain = s2z(s_zeros, s_poles, s_gain, fs)
        sos.append(zpk2sos(z_zeros, z_poles, z_gain, pairing='nearest'))
    selection = util.source_selection_all(len(x0))
    return delay, weight, sos, phaseshift, selection, secondary_source_point(c)


def plane_3d(x0, r0, npw, fs, max_order=None, c=None, s2z=matchedz_zpk):
    r"""Virtual plane wave by 3-dimensional NFC-HOA.

    .. math::

        D(\phi_0, s) =
        \frac{\e{\frac{s}{c}r_0}}{r_0}
        \sum_{n=0}^{N}
        (-1)^n (2n+1) P_{n}(\cos\Theta)
        \Big(\frac{s}{s-\frac{c}{r_0}\sigma_0}\Big)^\mu
        \prod_{l=1}^{\nu}
        \frac{s^2}{(s-\frac{c}{r_0}\sigma_l)^2+(\frac{c}{r_0}\omega_l)^2}

    The driving function is represented in the Laplace domain,
    from which the recursive filters are designed.
    :math:`\sigma_l + \i\omega_l` denotes the complex roots of
    the reverse Bessel polynomial.
    The number of second-order sections is
    :math:`\nu = \big\lfloor\tfrac{|m|}{2}\big\rfloor`,
    whereas the number of first-order section :math:`\mu` is either 0 or 1
    for even and odd :math:`|m|`, respectively.
    :math:`P_{n}(\cdot)` denotes the Legendre polynomial of degree :math:`n`,
    and :math:`\Theta` the angle between :math:`(\theta, \phi)`
    and :math:`(\theta_\text{pw}, \phi_\text{pw})`.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    r0 : float
        Radius of the spherical secondary source distribution.
    npw : (3,) array_like
        Unit vector (propagation direction) of plane wave.
    fs : int
        Sampling frequency in Hertz.
    max_order : int, optional
        Ambisonics order.
    c : float, optional
        Speed of sound in m/s.
    s2z : callable, optional
        Function transforming s-domain poles and zeros into z-domain,
        e.g. :func:`matchedz_zpk`, :func:`scipy.signal.bilinear_zpk`.

    Returns
    -------
    delay : float
        Overall delay in seconds.
    weight : float
        Overall weight.
    sos : list of numpy.ndarray
        Second-order section filters :func:`scipy.signal.sosfilt`.
    phaseshift : (N,) numpy.ndarray
        Phase shift in radians.
    selection : (N,) numpy.ndarray
        Boolean array containing only ``True`` indicating that
        all secondary source are "active" for NFC-HOA.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.time.synthesize()`.

    """
    if max_order is None:
        max_order = util.max_order_spherical_harmonics(len(x0))
    if c is None:
        c = default.c

    x0 = util.asarray_of_rows(x0)
    npw = util.asarray_1d(npw)
    phi0, theta0, _ = util.cart2sph(*x0.T)
    phipw, thetapw, _ = util.cart2sph(*npw)
    phaseshift = np.arccos(np.dot(x0 / r0, -npw))

    delay = -r0 / c
    weight = 4 * np.pi / r0
    sos = []
    for m in range(max_order + 1):
        _, p, _ = besselap(m, norm='delay')
        s_zeros = np.zeros(m)
        s_poles = c / r0 * p
        s_gain = 1
        z_zeros, z_poles, z_gain = s2z(s_zeros, s_poles, s_gain, fs)
        sos.append(zpk2sos(z_zeros, z_poles, z_gain, pairing='nearest'))
    selection = util.source_selection_all(len(x0))
    return delay, weight, sos, phaseshift, selection, secondary_source_point(c)


def point_3d(x0, r0, xs, fs, max_order=None, c=None, s2z=matchedz_zpk):
    r"""Virtual point source by 3-dimensional NFC-HOA.

    .. math::

        D(\phi_0, s) =
        \frac{\e{\frac{s}{c}(r_0-r_\text{s})}}{4 \pi r_0 r_\text{s}}
        \sum_{n=0}^{N}
        (2n+1) P_{n}(\cos\Theta)
        \Big(\frac{s-\frac{c}{r_\text{s}}\sigma_0}{s-\frac{c}{r_0}\sigma_0}\Big)^\mu
        \prod_{l=1}^{\nu}
        \frac{(s-\frac{c}{r_\text{s}}\sigma_l)^2-(\frac{c}{r_\text{s}}\omega_l)^2}
        {(s-\frac{c}{r_0}\sigma_l)^2+(\frac{c}{r_0}\omega_l)^2}

    The driving function is represented in the Laplace domain,
    from which the recursive filters are designed.
    :math:`\sigma_l + \i\omega_l` denotes the complex roots of
    the reverse Bessel polynomial.
    The number of second-order sections is
    :math:`\nu = \big\lfloor\tfrac{|m|}{2}\big\rfloor`,
    whereas the number of first-order section :math:`\mu` is either 0 or 1
    for even and odd :math:`|m|`, respectively.
    :math:`P_{n}(\cdot)` denotes the Legendre polynomial of degree :math:`n`,
    and :math:`\Theta` the angle between :math:`(\theta, \phi)`
    and :math:`(\theta_\text{s}, \phi_\text{s})`.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    r0 : float
        Radius of the spherial secondary source distribution.
    xs : (3,) array_like
        Virtual source position.
    fs : int
        Sampling frequency in Hertz.
    max_order : int, optional
        Ambisonics order.
    c : float, optional
        Speed of sound in m/s.
    s2z : callable, optional
        Function transforming s-domain poles and zeros into z-domain,
        e.g. :func:`matchedz_zpk`, :func:`scipy.signal.bilinear_zpk`.

    Returns
    -------
    delay : float
        Overall delay in seconds.
    weight : float
        Overall weight.
    sos : list of numpy.ndarray
        Second-order section filters :func:`scipy.signal.sosfilt`.
    phaseshift : (N,) numpy.ndarray
        Phase shift in radians.
    selection : (N,) numpy.ndarray
        Boolean array containing only ``True`` indicating that
        all secondary source are "active" for NFC-HOA.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.time.synthesize()`.

    """
    if max_order is None:
        max_order = util.max_order_spherical_harmonics(len(x0))
    if c is None:
        c = default.c

    x0 = util.asarray_of_rows(x0)
    xs = util.asarray_1d(xs)
    phi0, theta0, _ = util.cart2sph(*x0.T)
    phis, thetas, rs = util.cart2sph(*xs)
    phaseshift = np.arccos(np.dot(x0 / r0, xs / rs))

    delay = (rs - r0) / c
    weight = 1 / r0 / rs
    sos = []
    for m in range(max_order + 1):
        _, p, _ = besselap(m, norm='delay')
        s_zeros = c / rs * p
        s_poles = c / r0 * p
        s_gain = 1
        z_zeros, z_poles, z_gain = s2z(s_zeros, s_poles, s_gain, fs)
        sos.append(zpk2sos(z_zeros, z_poles, z_gain, pairing='nearest'))
    selection = util.source_selection_all(len(x0))
    return delay, weight, sos, phaseshift, selection, secondary_source_point(c)


def driving_signals_25d(delay, weight, sos, phaseshift, signal):
    """Get 2.5-dimensional NFC-HOA driving signals.

    Parameters
    ----------
    delay : float
        Overall delay in seconds.
    weight : float
        Overall weight.
    sos : list of array_like
        Second-order section filters :func:`scipy.signal.sosfilt`.
    phaseshift : (N,) array_like
        Phase shift in radians.
    signal : (L,) array_like + float
        Excitation signal consisting of (mono) audio data and a sampling
        rate (in Hertz).  A `DelayedSignal` object can also be used.

    Returns
    -------
    `DelayedSignal`
        A tuple containing the delayed signals (in a `numpy.ndarray`
        with shape ``(L, N)``), followed by the sampling rate (in Hertz)
        and a (possibly negative) time offset (in seconds).

    """
    data, fs, t_offset = util.as_delayed_signal(signal)
    N = len(phaseshift)
    out = np.tile(np.expand_dims(sosfilt(sos[0], data), 1), (1, N))
    for m in range(1, len(sos)):
        modal_response = sosfilt(sos[m], data)[:, np.newaxis]
        out += modal_response * np.cos(m * phaseshift)
    return util.DelayedSignal(2 * weight * out, fs, t_offset + delay)


def driving_signals_3d(delay, weight, sos, phaseshift, signal):
    """Get 3-dimensional NFC-HOA driving signals.

    Parameters
    ----------
    delay : float
        Overall delay in seconds.
    weight : float
        Overall weight.
    sos : list of array_like
        Second-order section filters :func:`scipy.signal.sosfilt`.
    phaseshift : (N,) array_like
        Phase shift in radians.
    signal : (L,) array_like + float
        Excitation signal consisting of (mono) audio data and a sampling
        rate (in Hertz).  A `DelayedSignal` object can also be used.

    Returns
    -------
    `DelayedSignal`
        A tuple containing the delayed signals (in a `numpy.ndarray`
        with shape ``(L, N)``), followed by the sampling rate (in Hertz)
        and a (possibly negative) time offset (in seconds).

    """
    data, fs, t_offset = util.as_delayed_signal(signal)
    N = len(phaseshift)
    out = np.tile(np.expand_dims(sosfilt(sos[0], data), 1), (1, N))
    for m in range(1, len(sos)):
        modal_response = sosfilt(sos[m], data)[:, np.newaxis]
        out += (2 * m + 1) * modal_response * legendre(m, np.cos(phaseshift))
    return util.DelayedSignal(weight / 4 / np.pi * out, fs, t_offset + delay)
