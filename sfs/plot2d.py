"""2D plots of sound fields etc."""
import matplotlib as _mpl
import matplotlib.pyplot as _plt
from mpl_toolkits import axes_grid1 as _axes_grid1
import numpy as _np

from . import default as _default
from . import util as _util


def _make_extreme(rgba):
    """Make bright colors darker, dark colors brighter, both less saturated."""
    # The package `colorspacious` must be installed for this to work:
    from colorspacious import cspace_convert
    lightness_step = 25
    chroma_factor = 0.7
    j, c, h = cspace_convert(rgba[:3], 'sRGB1', 'JCh')
    if j > 50:
        j -= lightness_step
    else:
        j += lightness_step
    c *= chroma_factor
    rgba[:3] = _np.clip(cspace_convert([j, c, h], 'JCh', 'sRGB1'), 0, 1)
    return rgba


def _register_cmap_with_extremes(name, original_name, **kwargs):
    """Create a color map with "under" and "over" values."""
    cmap = _plt.get_cmap(original_name)
    cmap = cmap.with_extremes(**kwargs)
    cmap.name = name
    _plt.colormaps.register(cmap=cmap)


# The following under/over values have been calculated with _make_extreme().
# They are hard-coded to avoid a dependency on the library "colorspacious".
_register_cmap_with_extremes('cividis_clip', 'cividis',
    under=[0.3581123750444155, 0.4308239004832521, 0.5431626919728758, 1.0],
    over=[0.748794386079359, 0.6952014568472878, 0.27380570592765713, 1.0])
_register_cmap_with_extremes('cividis_r_clip', 'cividis_r',
    under=[0.748794386079359, 0.6952014568472878, 0.27380570592765713, 1.0],
    over=[0.3581123750444155, 0.4308239004832521, 0.5431626919728758, 1.0])
_register_cmap_with_extremes('inferno_clip', 'inferno',
    under=[0.3223737972210511, 0.3196564508033573, 0.3474201893768059, 1.0],
    over=[0.7759331577663429, 0.7815136432099379, 0.5546145677840046, 1.0])
_register_cmap_with_extremes('inferno_r_clip', 'inferno_r',
    under=[0.7759331577663429, 0.7815136432099379, 0.5546145677840046, 1.0],
    over=[0.3223737972210511, 0.3196564508033573, 0.3474201893768059, 1.0])
_register_cmap_with_extremes('magma_clip', 'magma',
    under=[0.3223737972210511, 0.3196564508033573, 0.3474201893768059, 1.0],
    over=[0.7755917106347097, 0.7765145738617047, 0.621366899182334, 1.0])
_register_cmap_with_extremes('magma_r_clip', 'magma_r',
    under=[0.7755917106347097, 0.7765145738617047, 0.621366899182334, 1.0],
    over=[0.3223737972210511, 0.3196564508033573, 0.3474201893768059, 1.0])
_register_cmap_with_extremes('plasma_clip', 'plasma',
    under=[0.3425913695445096, 0.42529714344969144, 0.66039452922638, 1.0],
    over=[0.723897190872765, 0.7507494961114689, 0.2574503078804632, 1.0])
_register_cmap_with_extremes('plasma_r_clip', 'plasma_r',
    under=[0.723897190872765, 0.7507494961114689, 0.2574503078804632, 1.0],
    over=[0.3425913695445096, 0.42529714344969144, 0.66039452922638, 1.0])
_register_cmap_with_extremes('viridis_clip', 'viridis',
    under=[0.536623905475994, 0.3775029064902613, 0.5655492658877974, 1.0],
    over=[0.7453792828268919, 0.6916769483054797, 0.24219807423955453, 1.0])
_register_cmap_with_extremes('viridis_r_clip', 'viridis_r',
    under=[0.7453792828268919, 0.6916769483054797, 0.24219807423955453, 1.0],
    over=[0.536623905475994, 0.3775029064902613, 0.5655492658877974, 1.0])

_register_cmap_with_extremes('RdBu_clip', 'RdBu',
    under=[0.6931181505544421, 0.3643024937396605, 0.40355267940576434, 1.0],
    over=[0.3844199587527661, 0.4705632423745359, 0.597949800233206, 1.0])
_register_cmap_with_extremes('RdBu_r_clip', 'RdBu_r',
    under=[0.3844199587527661, 0.4705632423745359, 0.597949800233206, 1.0],
    over=[0.6931181505544421, 0.3643024937396605, 0.40355267940576434, 1.0])

# The 'coolwarm' colormap is based on the paper
# "Diverging Color Maps for Scientific Visualization" by Kenneth Moreland
# https://www.kennethmoreland.com/color-maps/ColorMapsExpanded.pdf
_register_cmap_with_extremes('coolwarm_clip', 'coolwarm',
    under=[0.510515040101537, 0.5812838578665308, 0.8594377816482693, 1.0],
    over=[0.9464304571740522, 0.43619922642510556, 0.4340300540797168, 1.0])
_register_cmap_with_extremes('coolwarm_r_clip', 'coolwarm_r',
    under=[0.9464304571740522, 0.43619922642510556, 0.4340300540797168, 1.0],
    over=[0.510515040101537, 0.5812838578665308, 0.8594377816482693, 1.0])


def _register_cmap_transparent(name, color):
    """Create a color map from a given color to transparent."""
    from matplotlib.colors import colorConverter, LinearSegmentedColormap
    red, green, blue = colorConverter.to_rgb(color)
    cdict = {'red': ((0, red, red), (1, red, red)),
             'green': ((0, green, green), (1, green, green)),
             'blue': ((0, blue, blue), (1, blue, blue)),
             'alpha': ((0, 0, 0), (1, 1, 1))}
    cmap = LinearSegmentedColormap(name, cdict)
    _plt.colormaps.register(cmap=cmap)


_register_cmap_transparent('blacktransparent', 'black')


def virtualsource(xs, ns=None, type='point', *, ax=None):
    """Draw position/orientation of virtual source."""
    xs = _np.asarray(xs)
    ns = _np.asarray(ns)
    if ax is None:
        ax = _plt.gca()

    if type == 'point':
        vps = _plt.Circle(xs, .05, edgecolor='k', facecolor='k')
        ax.add_artist(vps)
        for n in range(1, 3):
            vps = _plt.Circle(xs, .05+n*0.05, edgecolor='k', fill=False)
            ax.add_artist(vps)
    elif type == 'plane':
        ns = 0.2 * ns

        ax.arrow(xs[0], xs[1], ns[0], ns[1], head_width=0.05,
                 head_length=0.1, fc='k', ec='k')


def reference(xref, *, size=0.1, ax=None):
    """Draw reference/normalization point."""
    xref = _np.asarray(xref)
    if ax is None:
        ax = _plt.gca()

    ax.plot((xref[0]-size, xref[0]+size), (xref[1]-size, xref[1]+size), 'k-')
    ax.plot((xref[0]-size, xref[0]+size), (xref[1]+size, xref[1]-size), 'k-')


def secondary_sources(x0, n0, *, size=0.05, grid=None):
    """Simple visualization of secondary source locations.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Loudspeaker positions.
    n0 : (N, 3) or (3,) array_like
        Normal vector(s) of loudspeakers.
    size : float, optional
        Size of loudspeakers in metres.
    grid : triple of array_like, optional
        If specified, only loudspeakers within the *grid* are shown.
    """
    x0 = _np.asarray(x0)
    n0 = _np.asarray(n0)
    ax = _plt.gca()

    # plot only secondary sources inside simulated area
    if grid is not None:
        x0, n0 = _visible_secondarysources(x0, n0, grid)

    # plot symbols
    for x00 in x0:
        ss = _plt.Circle(x00[0:2], size, edgecolor='k', facecolor='k')
        ax.add_artist(ss)


def loudspeakers(x0, n0, a0=0.5, *, size=0.08, show_numbers=False, grid=None,
                 ax=None, zorder=2):
    """Draw loudspeaker symbols at given locations and angles.

    The default ``zorder`` is changed to 2, which is the same as line plots
    (e.g. `level_contour()`).

    Parameters
    ----------
    x0 : (N, 3) array_like
        Loudspeaker positions.
    n0 : (N, 3) or (3,) array_like
        Normal vector(s) of loudspeakers.
    a0 : float or (N,) array_like, optional
        Weighting factor(s) of loudspeakers.
    size : float, optional
        Size of loudspeakers in metres.
    show_numbers : bool, optional
        If ``True``, loudspeaker numbers are shown.
    grid : triple of array_like, optional
        If specified, only loudspeakers within the *grid* are shown.
    ax : Axes object, optional
        The loudspeakers are plotted into this `matplotlib.axes.Axes`
        object or -- if not specified -- into the current axes.

    """
    x0 = _util.asarray_of_rows(x0)
    n0 = _util.asarray_of_rows(n0)
    a0 = _util.asarray_1d(a0).reshape(-1, 1)

    # plot only secondary sources inside simulated area
    if grid is not None:
        x0, n0 = _visible_secondarysources(x0, n0, grid)

    # normalized coordinates of loudspeaker symbol (see IEC 60617-9)
    codes, coordinates = zip(*(
        (_mpl.path.Path.MOVETO, [-0.62, 0.21]),
        (_mpl.path.Path.LINETO, [-0.31, 0.21]),
        (_mpl.path.Path.LINETO, [0, 0.5]),
        (_mpl.path.Path.LINETO, [0, -0.5]),
        (_mpl.path.Path.LINETO, [-0.31, -0.21]),
        (_mpl.path.Path.LINETO, [-0.62, -0.21]),
        (_mpl.path.Path.CLOSEPOLY, [0, 0]),
        (_mpl.path.Path.MOVETO, [-0.31, 0.21]),
        (_mpl.path.Path.LINETO, [-0.31, -0.21]),
    ))
    coordinates = _np.column_stack([coordinates, _np.zeros(len(coordinates))])
    coordinates *= size

    patches = []
    for x00, n00 in _util.broadcast_zip(x0, n0):
        # rotate and translate coordinates
        R = _util.rotation_matrix([1, 0, 0], n00)
        transformed_coordinates = _np.inner(coordinates, R) + x00

        patches.append(_mpl.patches.PathPatch(_mpl.path.Path(
            transformed_coordinates[:, :2], codes)))

    # add collection of patches to current axis
    p = _mpl.collections.PatchCollection(
        patches, edgecolor='0', facecolor=_np.tile(1 - a0, 3), zorder=zorder)
    if ax is None:
        ax = _plt.gca()
    ax.add_collection(p)

    if show_numbers:
        for idx, (x00, n00) in enumerate(_util.broadcast_zip(x0, n0)):
            x, y = x00[:2] - 1.2 * size * n00[:2]
            ax.text(x, y, idx + 1, horizontalalignment='center',
                    verticalalignment='center', clip_on=True)


def _visible_secondarysources(x0, n0, grid):
    """Determine secondary sources which lie within *grid*."""
    x, y = _util.as_xyz_components(grid[:2])
    idx = _np.where((x0[:, 0] > x.min()) & (x0[:, 0] < x.max()) &
                    (x0[:, 1] > y.min()) & (x0[:, 1] < x.max()))
    idx = _np.squeeze(idx)

    return x0[idx, :], n0[idx, :]


def _plotting_plane(p, grid):
    if p.ndim == 3:
        if p.shape[2] == 1:
            p = p[:, :, 0]  # first axis: y; second axis: x
            plotting_plane = 'xy'
        elif p.shape[1] == 1:
            p = p[:, 0, :].T  # first axis: z; second axis: y
            plotting_plane = 'yz'
        elif p.shape[0] == 1:
            p = p[0, :, :].T  # first axis: z; second axis: x
            plotting_plane = 'xz'
        else:
            raise ValueError("If p is 3D, one dimension must have length 1")
    elif len(grid) == 3:
        if grid[2].ndim == 0:
            plotting_plane = 'xy'
        elif grid[1].ndim == 0:
            plotting_plane = 'xz'
        elif grid[0].ndim == 0:
            plotting_plane = 'yz'
        else:
            raise ValueError(
                "If p is 2D and grid is 3D, one grid component must be scalar")
    else:
        # 2-dimensional case
        plotting_plane = 'xy'

    if plotting_plane == 'xy':
        x, y = grid[[0, 1]]
    elif plotting_plane == 'xz':
        x, y = grid[[0, 2]]
    elif plotting_plane == 'yz':
        x, y = grid[[1, 2]]

    dx = 0.5 * _np.ptp(x) / p.shape[0]
    dy = 0.5 * _np.ptp(y) / p.shape[1]

    extent = x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy

    return p, extent, plotting_plane


def amplitude(p, grid, *, xnorm=None, cmap='coolwarm_clip',
              vmin=-2.0, vmax=2.0, xlabel=None, ylabel=None,
              colorbar=True, colorbar_kwargs=None, ax=None, **kwargs):
    """Two-dimensional plot of sound field (real part).

    Parameters
    ----------
    p : array_like
        Sound pressure values (or any other scalar quantity if you
        like).  If the values are complex, the imaginary part is
        ignored.
        Typically, *p* is two-dimensional with a shape of *(Ny, Nx)*,
        *(Nz, Nx)* or *(Nz, Ny)*.  This is the case if
        `sfs.util.xyz_grid()` was used with a single number for *z*,
        *y* or *x*, respectively.
        However, *p* can also be three-dimensional with a shape of *(Ny,
        Nx, 1)*, *(1, Nx, Nz)* or *(Ny, 1, Nz)*.  This is the case if
        :func:`numpy.meshgrid` was used with a scalar for *z*, *y* or
        *x*, respectively (and of course with the default
        ``indexing='xy'``).

        .. note:: If you want to plot a single slice of a pre-computed
                  "full" 3D sound field, make sure that the slice still
                  has three dimensions (including one singleton
                  dimension).  This way, you can use the original *grid*
                  of the full volume without changes.
                  This works because the grid component corresponding to
                  the singleton dimension is simply ignored.

    grid : triple or pair of numpy.ndarray
        The grid that was used to calculate *p*, see
        `sfs.util.xyz_grid()`.  If *p* is two-dimensional, but
        *grid* has 3 components, one of them must be scalar.
    xnorm : array_like, optional
        Coordinates of a point to which the sound field should be
        normalized before plotting.  If not specified, no normalization
        is used.  See `sfs.util.normalize()`.

    Returns
    -------
    AxesImage
        See :func:`matplotlib.pyplot.imshow`.

    Other Parameters
    ----------------
    xlabel, ylabel : str
        Overwrite default x/y labels.  Use ``xlabel=''`` and
        ``ylabel=''`` to remove x/y labels.  The labels can be changed
        afterwards with :func:`matplotlib.pyplot.xlabel` and
        :func:`matplotlib.pyplot.ylabel`.
    colorbar : bool, optional
        If ``False``, no colorbar is created.
    colorbar_kwargs : dict, optional
        Further colorbar arguments, see `add_colorbar()`.
    ax : Axes, optional
        If given, the plot is created on *ax* instead of the current
        axis (see :func:`matplotlib.pyplot.gca`).
    cmap, vmin, vmax, **kwargs
        All further parameters are forwarded to
        :func:`matplotlib.pyplot.imshow`.

    See Also
    --------
    sfs.plot2d.level

    """
    p = _np.asarray(p)
    grid = _util.as_xyz_components(grid)

    # normalize sound field wrt xnorm
    if xnorm is not None:
        p = _util.normalize(p, grid, xnorm)

    p, extent, plotting_plane = _plotting_plane(p, grid)

    if ax is None:
        ax = _plt.gca()

    # see https://github.com/matplotlib/matplotlib/issues/10567
    if _mpl.__version__.startswith('2.1.'):
        p = _np.clip(p, -1e15, 1e15)  # clip to float64 range

    im = ax.imshow(_np.real(p), cmap=cmap, origin='lower',
                   extent=extent, vmax=vmax, vmin=vmin, **kwargs)
    if xlabel is None:
        xlabel = plotting_plane[0] + ' / m'
    if ylabel is None:
        ylabel = plotting_plane[1] + ' / m'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if colorbar:
        if colorbar_kwargs is None:
            colorbar_kwargs = dict(extend='both')
        add_colorbar(im, **colorbar_kwargs)
    return im


def level(p, grid, *, xnorm=None, power=False, cmap='viridis_clip',
          vmax=3, vmin=-50, colorbar_kwargs=None, **kwargs):
    """Two-dimensional plot of level (dB) of sound field.

    Takes the same parameters as `sfs.plot2d.amplitude()`.

    Other Parameters
    ----------------
    power : bool, optional
        See `sfs.util.db()`.

    """
    # normalize before converting to dB!
    if xnorm is not None:
        p = _util.normalize(p, grid, xnorm)
    L = _util.db(p, power=power)
    if colorbar_kwargs is None:
        colorbar_kwargs = dict(extend='both', label='level / dB')
    return amplitude(L, grid=grid, xnorm=None, cmap=cmap,
                     vmax=vmax, vmin=vmin,
                     colorbar_kwargs=colorbar_kwargs, **kwargs)


def level_contour(p, grid, *, xnorm=None, power=False,
                  xlabel=None, ylabel=None, ax=None, **kwargs):
    """Two-dimensional contour plot of level (dB) of sound field.

    Parameters
    ----------
    p, grid, xnorm, power, xlabel, ylabel, ax
        Same as in `level()`.
    **kwargs
        All further parameters are forwarded to
        :func:`matplotlib.pyplot.contour`.

    """
    p = _np.asarray(p)
    grid = _util.as_xyz_components(grid)
    # normalize before converting to dB!
    if xnorm is not None:
        p = _util.normalize(p, grid, xnorm)
    p, extent, plotting_plane = _plotting_plane(p, grid)
    L = _util.db(p, power=power)
    if ax is None:
        ax = _plt.gca()
    contour = ax.contour(L, extent=extent, **kwargs)
    if xlabel is None:
        xlabel = plotting_plane[0] + ' / m'
    if ylabel is None:
        ylabel = plotting_plane[1] + ' / m'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return contour


def particles(x, *, trim=None, ax=None, xlabel='x (m)', ylabel='y (m)',
              edgecolors=None, marker='.', s=15, **kwargs):
    """Plot particle positions as scatter plot.

    Parameters
    ----------
    x : triple or pair of array_like
        x, y and optionally z components of particle positions. The z
        components are ignored.
        If the values are complex, the imaginary parts are ignored.

    Returns
    -------
    Scatter
        See :func:`matplotlib.pyplot.scatter`.

    Other Parameters
    ----------------
    trim : array of float, optional
        xmin, xmax, ymin, ymax limits for which the particles are plotted.
    ax : Axes, optional
        If given, the plot is created on *ax* instead of the current
        axis (see :func:`matplotlib.pyplot.gca`).
    xlabel, ylabel : str
        Overwrite default x/y labels.  Use ``xlabel=''`` and
        ``ylabel=''`` to remove x/y labels.  The labels can be changed
        afterwards with :func:`matplotlib.pyplot.xlabel` and
        :func:`matplotlib.pyplot.ylabel`.
    edgecolors, markr, s, **kwargs
        All further parameters are forwarded to
        :func:`matplotlib.pyplot.scatter`.

    """
    XX, YY = [_np.real(c) for c in x[:2]]

    if trim is not None:
        xmin, xmax, ymin, ymax = trim

        idx = _np.where((XX > xmin) & (XX < xmax) & (YY > ymin) & (YY < ymax))
        XX = XX[idx]
        YY = YY[idx]

    if ax is None:
        ax = _plt.gca()

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax.scatter(XX, YY, edgecolors=edgecolors, marker=marker, s=s,
                      **kwargs)


def vectors(v, grid, *, cmap='blacktransparent', headlength=3,
            headaxislength=2.5, ax=None, clim=None, **kwargs):
    """Plot a vector field in the xy plane.

    Parameters
    ----------
    v : triple or pair of array_like
        x, y and optionally z components of vector field.  The z
        components are ignored.
        If the values are complex, the imaginary parts are ignored.
    grid : triple or pair of array_like
        The grid that was used to calculate *v*, see
        `sfs.util.xyz_grid()`.  Any z components are ignored.

    Returns
    -------
    Quiver
        See :func:`matplotlib.pyplot.quiver`.

    Other Parameters
    ----------------
    ax : Axes, optional
        If given, the plot is created on *ax* instead of the current
        axis (see :func:`matplotlib.pyplot.gca`).
    clim : pair of float, optional
        Limits for the scaling of arrow colors.
        See :func:`matplotlib.pyplot.quiver`.
    cmap, headlength, headaxislength, **kwargs
        All further parameters are forwarded to
        :func:`matplotlib.pyplot.quiver`.

    """
    v = _util.as_xyz_components(v[:2]).apply(_np.real)
    X, Y = _util.as_xyz_components(grid[:2])
    speed = _np.linalg.norm(v)
    with _np.errstate(invalid='ignore'):
        U, V = v.apply(_np.true_divide, speed)
    if ax is None:
        ax = _plt.gca()
    if clim is None:
        v_ref = 1 / (_default.rho0 * _default.c)  # reference particle velocity
        clim = 0, 2 * v_ref
    return ax.quiver(X, Y, U, V, speed, cmap=cmap, pivot='mid', units='xy',
                     angles='xy', headlength=headlength,
                     headaxislength=headaxislength, clim=clim, **kwargs)


def add_colorbar(im, *, aspect=20, pad=0.5, **kwargs):
    r"""Add a vertical color bar to a plot.

    Parameters
    ----------
    im : ScalarMappable
        The output of `sfs.plot2d.amplitude()`, `sfs.plot2d.level()` or any
        other `matplotlib.cm.ScalarMappable`.
    aspect : float, optional
        Aspect ratio of the colorbar.  Strictly speaking, since the
        colorbar is vertical, it's actually the inverse of the aspect
        ratio.
    pad : float, optional
        Space between image plot and colorbar, as a fraction of the
        width of the colorbar.

        .. note:: The *pad* argument of
                  :meth:`matplotlib.figure.Figure.colorbar` has a
                  slightly different meaning ("fraction of original
                  axes")!
    \**kwargs
        All further arguments are forwarded to
        :meth:`matplotlib.figure.Figure.colorbar`.

    See Also
    --------
    matplotlib.pyplot.colorbar

    """
    ax = im.axes
    divider = _axes_grid1.make_axes_locatable(ax)
    width = _axes_grid1.axes_size.AxesY(ax, aspect=1/aspect)
    pad = _axes_grid1.axes_size.Fraction(pad, width)
    current_ax = _plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    _plt.sca(current_ax)
    return ax.figure.colorbar(im, cax=cax, orientation='vertical', **kwargs)
