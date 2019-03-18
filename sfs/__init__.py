"""Sound Field Synthesis Toolbox.

https://sfs-python.readthedocs.io/

.. rubric:: Submodules

.. autosummary::
    :toctree:

    fd
    td
    array
    tapering
    plot2d
    plot3d
    util

"""
__version__ = "0.5.0"


class default:
    """Get/set defaults for the *sfs* module.

    For example, when you want to change the default speed of sound::

        import sfs
        sfs.default.c = 330

    """

    c = 343
    """Speed of sound."""

    rho0 = 1.2250
    """Static density of air."""

    selection_tolerance = 1e-6
    """Tolerance used for secondary source selection."""

    def __setattr__(self, name, value):
        """Only allow setting existing attributes."""
        if name in dir(self) and name != 'reset':
            super().__setattr__(name, value)
        else:
            raise AttributeError(
                '"default" object has no attribute ' + repr(name))

    def reset(self):
        """Reset all attributes to their "factory default"."""
        vars(self).clear()


import sys as _sys
if not getattr(_sys.modules.get('sphinx'), 'SFS_DOCS_ARE_BEING_BUILT', False):
    # This object shadows the 'default' class, except when the docs are built:
    default = default()

from . import tapering
from . import array
from . import util
try:
    from . import plot2d
except ImportError:
    pass
try:
    from . import plot3d
except ImportError:
    pass

from . import fd
from . import td
