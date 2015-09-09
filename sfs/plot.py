"""Plot sound fields etc."""

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from . import util


def _register_coolwarm_clip():
    """Create color map with "over" and "under" values."""
    from matplotlib.colors import LinearSegmentedColormap
    # The 'coolwarm' colormap is based on the paper
    # "Diverging Color Maps for Scientific Visualization" by Kenneth Moreland
    # http://www.sandia.gov/~kmorel/documents/ColorMaps/
    cdict = plt.cm.datad['coolwarm']
    cmap = LinearSegmentedColormap('coolwarm_clip', cdict)
    cmap.set_over(color=cmap(1.0), alpha=0.7)
    cmap.set_under(color=cmap(0.0), alpha=0.7)
    plt.cm.register_cmap(cmap=cmap)

_register_coolwarm_clip()
del _register_coolwarm_clip


def virtualsource_2d(xs, ns=None, type='point', ax=None):
    """Draw position/orientation of virtual source."""
    xs = np.asarray(xs)
    ns = np.asarray(ns)
    if ax is None:
        ax = plt.axes()

    if type == 'point':
        vps = plt.Circle(xs, .05, edgecolor='k', facecolor='k')
        ax.add_artist(vps)
        for n in range(1, 3):
            vps = plt.Circle(xs, .05+n*0.05, edgecolor='k', fill=False)
            ax.add_artist(vps)
    elif type == 'plane':
        ns = 0.2 * ns

        ax.arrow(xs[0], xs[1], ns[0], ns[1], head_width=0.05,
                 head_length=0.1, fc='k', ec='k')


def reference_2d(xref, size=0.1, ax=None):
    """Draw reference/normalization point."""
    xref = np.asarray(xref)
    if ax is None:
        ax = plt.axes()

    ax.plot((xref[0]-size, xref[0]+size), (xref[1]-size, xref[1]+size), 'k-')
    ax.plot((xref[0]-size, xref[0]+size), (xref[1]+size, xref[1]-size), 'k-')


def secondarysource_2d(x0, n0, grid=None):
    """Simple plot of secondary source locations."""
    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    ax = plt.axes()

    # plot only secondary sources inside simulated area
    if grid is not None:
        x0, n0 = _visible_secondarysources_2d(x0, n0, grid)

    # plot symbols
    for x00 in x0:
        ss = plt.Circle(x00[0:2], .05, edgecolor='k', facecolor='k')
        ax.add_artist(ss)


def loudspeaker_2d(x0, n0, a0=0.5, size=0.08, show_numbers=False, grid=None,
                   ax=None):
    """Draw loudspeaker symbols at given locations and angles.

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
    grid : triple of numpy.ndarray, optional
        If specified, only loudspeakers within the `grid` are shown.
    ax : Axes object, optional
        The loudspeakers are plotted into this
        :class:`~matplotlib.axes.Axes` object or -- if not specified --
        into the current axes.

    """
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    a0 = util.asarray_1d(a0).reshape(-1, 1)

    # plot only secondary sources inside simulated area
    if grid is not None:
        x0, n0 = _visible_secondarysources_2d(x0, n0, grid)

    # normalized coordinates of loudspeaker symbol (see IEC 60617-9)
    codes, coordinates = zip(*(
        (Path.MOVETO, [-0.62, 0.21]),
        (Path.LINETO, [-0.31, 0.21]),
        (Path.LINETO, [0, 0.5]),
        (Path.LINETO, [0, -0.5]),
        (Path.LINETO, [-0.31, -0.21]),
        (Path.LINETO, [-0.62, -0.21]),
        (Path.CLOSEPOLY, [0, 0]),
        (Path.MOVETO, [-0.31, 0.21]),
        (Path.LINETO, [-0.31, -0.21]),
    ))
    coordinates = np.column_stack([coordinates, np.zeros(len(coordinates))])
    coordinates *= size

    patches = []
    for x00, n00 in util.broadcast_zip(x0, n0):
        # rotate and translate coordinates
        R = util.rotation_matrix([1, 0, 0], n00)
        transformed_coordinates = np.inner(coordinates, R) + x00

        patches.append(PathPatch(Path(transformed_coordinates[:, :2], codes)))

    # add collection of patches to current axis
    p = PatchCollection(patches, edgecolor='0', facecolor=np.tile(1 - a0, 3))
    if ax is None:
        ax = plt.gca()
    ax.add_collection(p)

    if show_numbers:
        for idx, (x00, n00) in enumerate(util.broadcast_zip(x0, n0)):
            x, y = x00[:2] - 1.2 * size * n00[:2]
            ax.text(x, y, idx + 1, horizontalalignment='center',
                    verticalalignment='center')


def _visible_secondarysources_2d(x0, n0, grid):
    """Determine secondary sources which lie within `grid`."""
    grid = util.asarray_of_arrays(grid)
    x, y = grid[:2]
    idx = np.where((x0[:, 0] > x.min()) & (x0[:, 0] < x.max()) &
                   (x0[:, 1] > y.min()) & (x0[:, 1] < x.max()))
    idx = np.squeeze(idx)

    return x0[idx, :], n0[idx, :]


def loudspeaker_3d(x0, n0, a0=None, w=0.08, h=0.08):
    """Plot positions and normals of a 3D secondary source distribution."""
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x0[:, 0], x0[:, 1], x0[:, 2], n0[:, 0],
              n0[:, 1], n0[:, 2], length=0.1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Secondary Sources')
    fig.show()


def soundfield(p, grid, xnorm=None, colorbar=True, cmap='coolwarm_clip',
               ax=None, xlabel='x (m)', ylabel='y (m)', vmax=2.0, vmin=-2.0,
               **kwargs):
    """Two-dimensional plot of sound field."""
    grid = util.asarray_of_arrays(grid)

    # normalize sound field wrt xnorm
    if xnorm is not None:
        p = util.normalize(p, grid, xnorm)

    x, y = grid[:2]  # ignore z-component

    if ax is None:
        ax = plt.gca()

    im = ax.imshow(np.real(p), cmap=cmap, origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   vmax=vmax, vmin=vmin, aspect='equal', **kwargs)
    ax.set_adjustable('box-forced')  # avoid empty space btw. axis and image
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if colorbar:
        ax.figure.colorbar(im, ax=ax)
    return im


def level(p, grid, xnorm=None, colorbar=True, cmap='coolwarm_clip',
          ax=None, xlabel='x (m)', ylabel='y (m)', vmax=3.0, vmin=-50,
          **kwargs):
    """Two-dimensional plot of level (dB) of sound field."""
    # normalize sound field wrt xnorm
    if xnorm is not None:
        p = util.normalize(p, grid, xnorm)

    xnorm = None
    im = soundfield(20*np.log10(np.abs(p)), grid, xnorm, colorbar, cmap, ax,
                    xlabel, ylabel, vmax, vmin, **kwargs)

    return im
