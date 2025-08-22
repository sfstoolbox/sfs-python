Installation
============

Requirements
------------

Obviously, you'll need Python_.
There are many ways to install Python,
and you can use any way you like,
however, we recommend using uv_ as shown in the steps below.

You can install ``uv`` with your favorite package manager,
or by one of the other methods described at
https://docs.astral.sh/uv/getting-started/installation/.

.. _Python: https://www.python.org/
.. _uv: https://docs.astral.sh/uv/
.. _NumPy: http://www.numpy.org/
.. _SciPy: https://www.scipy.org/scipylib/
.. _Matplotlib: https://matplotlib.org/

Installation
------------

First, create a new directory wherever you want, change into it, then run::

    uv venv

This will print instructions for how to `activate the environment`__.
You should do that now!

__ https://docs.astral.sh/uv/pip/environments/#using-a-virtual-environment

The Sound Field Synthesis Toolbox can be installed with::

    uv pip install sfs

This will automatically install the NumPy_ and SciPy_ libraries as well,
which are needed by the SFS Toolbox.

If you want to use the provided functions for plotting sound fields, you'll need
Matplotlib_::

    uv pip install matplotlib

However, since all results are provided as plain NumPy_ arrays, you should also
be able to use any other plotting library of your choice to visualize the sound
fields.

The steps above need to be executed only once.
Whenever you come back to this directory at a later time,
you just need to activate the environment again.

If you want to install the latest development version of the SFS Toolbox, have a
look at :doc:`contributing`.
