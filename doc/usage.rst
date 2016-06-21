Usage
=====

Requirements
------------

Obviously, you'll need Python_.
We normally use Python 3.x, but it *should* also work with Python 2.x.
NumPy_ and SciPy_ are needed for the calculations.
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
to download and install the latest release of the Sound Field Synthesis Toolbox
with a single command::

    python3 -m pip install sfs --user

If you want to install it system-wide for all users (assuming you have the
necessary rights), you can just drop the ``--user`` option.

To un-install, use::

    python3 -m pip uninstall sfs

.. _pip: http://www.pip-installer.org/en/latest/installing.html

How to Get Started
------------------

Various examples are located in the directory examples/

* sound_field_synthesis.py:
    Illustrates the general usage of the toolbox
* horizontal_plane_arrays.py:
    Computes the sound fields for various techniques, virtual sources and loudspeaker array configurations
* soundfigures.py:
    Illustrates the synthesis of sound figures with Wave Field Synthesis
