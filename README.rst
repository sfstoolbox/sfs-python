Sound Field Synthesis Toolbox for Python
========================================

Python implementation of the `Sound Field Synthesis Toolbox`_.

.. _Sound Field Synthesis Toolbox: http://github.com/sfstoolbox/sfs/

Documentation:
   http://sfs.rtfd.org/

Code:
   http://github.com/sfstoolbox/sfs-python/

Python Package Index:
   http://pypi.python.org/pypi/sfs/

Requirements
------------

Obviously, you'll need Python_.
We normally use Python 3.x, but it *should* also work with Python 2.x.
NumPy_ and SciPy_ are needed for the calcuations.
If you also want to plot the resulting sound fields, you'll need matplotlib_.

Instead of installing all of them separately, you should probably get a Python
distribution that already includes everything, e.g. Anaconda_.

.. _Python: http://www.python.org/
.. _NumPy: http://www.numpy.org/
.. _SciPy: http://www.scipy.org/scipylib/
.. _matplotlib: http://matplotlib.org/
.. _Anaconda: http://docs.continuum.io/anaconda/

Installation
------------

Once you have installed the above-mentioned dependencies, you can use pip_
to download and install the latest release with a single command::

   pip install sfs --user

If you want to install it system-wide for all users (assuming you have the
necessary rights), you can just drop the ``--user`` option.

To un-install, use::

   pip uninstall sfs

If you want to keep up with the latest and greatest development version you
should get it from Github_::

   git clone https://github.com/sfstoolbox/sfs-python.git
   cd sfs-python
   python setup.py develop --user

.. _pip: http://www.pip-installer.org/en/latest/installing.html
.. _Github: http://github.com/sfstoolbox/sfs-python/

How to Get Started
------------------

Various examples are located in the directory examples/

* sound_field_synthesis.py: 
    Illustrates the general usage of the toolbox
* horizontal_plane_arrays.py: 
    Computes the sound fields for various techniques, virtual sources and loudspeaker array configurations
* soundfigures.py: 
    Illustrates the synthesis of sound figures with Wave Field Synthesis
