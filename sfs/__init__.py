"""Sound Field Synthesis Toolbox.

http://sfs.rtfd.org/

"""
__version__ = "0.4.0"

from . import tapering
from . import array
from . import util
from . import defs
try:
    from . import plot
except ImportError:
    pass

from . import mono
from . import time
