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

If you don't like ``uv``, no problem!
You can also use Python's official packaging tool pip_ or any other third-party tool,
as long as it can install `the SFS package`_.

.. _Python: https://www.python.org/
.. _uv: https://docs.astral.sh/uv/
.. _pip: https://packaging.python.org/en/latest/tutorials/installing-packages/
.. _the SFS package: https://pypi.org/project/sfs/
.. _NumPy: http://www.numpy.org/
.. _SciPy: https://www.scipy.org/scipylib/
.. _Matplotlib: https://matplotlib.org/

Installation
------------

First, create a new directory wherever you want, change into it, then run::

    uv init --bare

This will create a file named ``pyproject.toml`` for you.
Use the ``--help`` flag to see other options.

The Sound Field Synthesis Toolbox can now be installed with::

    uv add sfs

This will automatically install the NumPy_ and SciPy_ libraries as well,
which are needed by the SFS Toolbox.
It will also create a file named ``uv.lock``, which tracks the exact versions
of all installed packages.

If you want to use the provided functions for plotting sound fields, you'll need
Matplotlib_::

    uv add matplotlib

However, since all results are provided as plain NumPy_ arrays, you should also
be able to use any other plotting library of your choice to visualize the sound
fields.

You might also want to install some other Python-related tools,
e.g. JupyterLab_::

    uv add jupyterlab

.. _JupyterLab: https://jupyter.org/

You get the gist: whatever you need, just ``uv add ...`` it!

Once everything is installed, you can start working with the tool of your choice
by simply prefixing it with ``uv run``, for example::

    uv run jupyter lab

Similarly, you can launch any other tool, like a text editor, an IDE etc.

You can also simply create a Python file, let's say ``my_script.py``:

.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np
    import sfs

    npw = sfs.util.direction_vector(np.radians(-45))
    f = 300  # Hz
    omega = 2 * np.pi * f

    grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.02)
    array = sfs.array.circular(N=32, R=1.5)

    d, selection, secondary_source = sfs.fd.wfs.plane_25d(
        omega, array.x, array.n, npw)

    p = sfs.fd.synthesize(d, selection, array, secondary_source, grid=grid)
    sfs.plot2d.amplitude(p, grid)
    sfs.plot2d.loudspeakers(array.x, array.n, selection * array.a, size=0.15)

    plt.show()

You can then run this script (assuming you installed ``matplotlib`` before) with::

    uv run my_script.py

In a similar way, you can run the :doc:`example-python-scripts`.

If you want to install the latest development version of the SFS Toolbox, have a
look at :doc:`contributing`.
