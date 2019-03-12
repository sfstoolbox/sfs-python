"""Submodules for monochromatic sound fields.

.. autosummary::
    :toctree:

    esaedge
    nfchoa
    sdm
    wfs

    source
    soundfigure

"""
import numpy as _np


from . import esaedge
from . import nfchoa
from . import sdm
from . import wfs

from . import source
from . import soundfigure

from .. import array as _array


def shiftphase(p, phase):
    """Shift phase of a sound field."""
    p = _np.asarray(p)
    return p * _np.exp(1j * phase)


def synthesize(d, weights, ssd, secondary_source_function, **kwargs):
    """Compute sound field for a generic driving function.

    Parameters
    ----------
    d : array_like
        Driving function.
    weights : array_like
        Additional weights applied during integration, e.g. source
        selection and tapering.
    ssd : sequence of between 1 and 3 array_like objects
        Positions, normal vectors and weights of secondary sources.
        A `SecondarySourceDistribution` can also be used.
    secondary_source_function : callable
        A function that generates the sound field of a secondary source.
        This signature is expected::

            secondary_source_function(
                position, normal_vector, weight, driving_function_weight,
                **kwargs) -> numpy.ndarray

    **kwargs
        All keyword arguments are forwarded to *secondary_source_function*.
        This is typically used to pass the *grid* argument.

    """
    ssd = _array.as_secondary_source_distribution(ssd)
    if not (len(ssd.x) == len(ssd.n) == len(ssd.a) == len(d) ==
            len(weights)):
        raise ValueError("length mismatch")
    p = 0
    for x, n, a, d, weight in zip(ssd.x, ssd.n, ssd.a, d, weights):
        if weight != 0:
            p += a * weight * d * secondary_source_function(x, n, **kwargs)
    return p
