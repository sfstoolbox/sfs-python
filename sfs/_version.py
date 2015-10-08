"""Version of the SFS Toolbox.

This is the only place where the version number is stored.
During installation, this file is read by ../setup.py and when importing
the 'sfs' module, this module is imported by __init__.py.

Whenever the version is incremented, a Git tag with the same name should
be created.

"""

__version__ = "0.1.1"
