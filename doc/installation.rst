Installation
============

Requirements
------------

Obviously, you'll need Python_.
More specifically, you'll need Python 3.
NumPy_ and SciPy_ are needed for the calculations.
If you want to use the provided functions for plotting sound fields, you'll need
Matplotlib_.
However, since all results are provided as plain NumPy_ arrays, you should also
be able to use any plotting library of your choice to visualize the sound
fields.

Instead of installing all of the requirements separately, you should probably
get a Python distribution that already includes everything, e.g. Anaconda_.

.. _Python: https://www.python.org/
.. _NumPy: http://www.numpy.org/
.. _SciPy: https://www.scipy.org/scipylib/
.. _Matplotlib: https://matplotlib.org/
.. _Anaconda: https://docs.anaconda.com/anaconda/

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

If you want to install the latest development version of the SFS Toolbox, have a
look at :doc:`contributing`.

.. _pip: https://pip.pypa.io/en/latest/installing/
