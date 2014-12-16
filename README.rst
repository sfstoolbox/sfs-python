Sound Field Synthesis Toolbox for Python
========================================

Python implementation of the `Sound Field Synthesis Toolbox`_.

.. _Sound Field Synthesis Toolbox: http://github.com/sfstoolbox/sfs/

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

Currently, the package is not yet available on PyPI_ (but coming soon!), for
now you should get it from Github_::

   git clone https://github.com/sfstoolbox/sfs-python.git
   cd sfs-python
   python setup.py develop --user

.. _PyPI: http://pypi.python.org/
.. _Github: http://github.com/sfstoolbox/sfs-python/
