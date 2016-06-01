"""Weights (tapering) for the driving function.

.. plot::
    :context: reset

    import sfs
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['figure.figsize'] = 8, 3  # inch
    plt.rcParams['axes.grid'] = True

    active1 = np.zeros(101, dtype=bool)
    active1[5:-5] = True

    # The active part can wrap around from the end to the beginning:
    active2 = np.ones(101, dtype=bool)
    active2[30:-10] = False

"""
import numpy as np


def none(active):
    """No tapering window.

    Examples
    --------
    .. plot::
        :context: close-figs

        plt.plot(sfs.tapering.none(active1))
        plt.axis([-3, 103, -0.1, 1.1])

    .. plot::
        :context: close-figs

        plt.plot(sfs.tapering.none(active2))
        plt.axis([-3, 103, -0.1, 1.1])

    """
    return active


def kaiser(active):
    """Kaiser tapering window.

    Examples
    --------
    .. plot::
        :context: close-figs

        plt.plot(sfs.tapering.kaiser(active1))
        plt.axis([-3, 103, -0.1, 1.1])

    .. plot::
        :context: close-figs

        plt.plot(sfs.tapering.kaiser(active2))
        plt.axis([-3, 103, -0.1, 1.1])

    """
    idx = _windowidx(active)
    window = np.zeros(len(active))
    window[idx] = np.kaiser(len(idx), 2)
    return window


def tukey(active, alpha):
    """Tukey tapering window.

    Examples
    --------
    .. plot::
        :context: close-figs

        plt.plot(sfs.tapering.tukey(active1, 0.1), label='α = 0.1')
        plt.plot(sfs.tapering.tukey(active1, 0.5), label='α = 0.5')
        plt.plot(sfs.tapering.tukey(active1, 0.75), label='α = 0.75')
        plt.plot(sfs.tapering.tukey(active1, 0.9), label='α = 0.9')
        plt.axis([-3, 103, -0.1, 1.1])
        plt.legend(loc='lower center')

    .. plot::
        :context: close-figs

        plt.plot(sfs.tapering.tukey(active2, 0.75))
        plt.axis([-3, 103, -0.1, 1.1])

    """
    idx = _windowidx(active)
    # alpha out of limits
    if alpha <= 0 or alpha >= 1:
        return none(active)
    # design Tukey window
    x = np.linspace(0, 1, len(idx)+2)
    w = np.ones_like(x)

    first_condition = x < alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha *
                                (x[first_condition] - alpha/2)))

    third_condition = x >= (1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha *
                                (x[third_condition] - 1 + alpha/2)))
    # fit window into tapering function
    window = np.zeros(len(active))
    window[idx] = w[1:-1]
    return window


def _windowidx(active):
    """Return list of connected indices for window function.

    Note: Gaps within the active part are not allowed.

    """
    # find index where active loudspeakers begin (works for connected contours)
    if (active[0] and not active[-1]) or np.all(active):
        first_idx = 0
    else:
        first_idx = np.argmax(np.diff(active.astype(int))) + 1
    # shift generic index vector to get a connected list of indices
    idx = np.roll(np.arange(len(active)), -first_idx)
    # remove indices of inactive secondary sources
    return idx[:np.count_nonzero(active)]
