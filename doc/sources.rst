Sources
=======

Initialization.

.. ipython:: python

   import numpy as np
   import sfs
   x0 = [1.5, 1, 0]
   f = 500
   omega = 2 * np.pi * f
   x = np.arange(-2, 2, 0.02)
   y = np.arange(-2, 2, 0.02)

Calculating the sound field.

.. ipython:: python

   p = sfs.mono.source.point(omega, x0, x, y)
   p

Plotting.

.. ipython:: python

   @savefig point_source.png width=50% align=center
   sfs.plot.soundfield(p, x, y)
