Different Types of Sound Sources
================================

This page shows some fundamental types of sound sources.
At the same time, it also shows how to calculate and plot sound fields.

First we have to import the ``sfs`` library.
Let's also import NumPy_ and matplotlib_, as we'll need them later.

.. _NumPy: http://www.numpy.org/
.. _matplotlib: http://matplotlib.org/

.. ipython:: python

   import sfs
   import numpy as np
   import matplotlib.pyplot as plt
   plt.rcParams['figure.figsize'] = 8, 4  # inch

We specify the position of the sound source in meters and its frequency in
Hertz, but we immediately convert this to an angular frequency in radians:

.. ipython:: python

   x0 = [1.5, 1, 0]
   f = 500  # Hz
   omega = 2 * np.pi * f

Now we create a spatial grid on which we will evaluate (and later plot) the
sound field.
For the x and y component, we give a minimum and maximum value, for the z
component we only specify a single scalar 0, i.e. we're interested in the
horizontal plane at :math:`z = 0`.
For more options, have a look at the documentation of
:func:`~sfs.util.xyz_grid`.

.. ipython:: python

   grid = sfs.util.xyz_grid([-2, 3], [-1, 2], 0, spacing=0.02)

Now we have all we need to calculate our first sound source.
Let's use the function :func:`sfs.mono.source.point` to create a
*point source*:

.. ipython:: python

   p_point = sfs.mono.source.point(omega, x0, None, grid)
   p_point.shape
   p_point

As we can see, the result is a two-dimensional array (because we asked for only
one z value) of complex numbers.

Those numbers look already quite impressive, but maybe it's even better if we
plot them?

.. ipython:: python

   sfs.plot.soundfield(p_point, grid);
   @savefig point_source_weak.png width=50% align=center
   plt.title("Point Source at {} m".format(x0));

Normalization ... multiply by :math:`4\pi` ...

.. ipython:: python

   p_point *= 4 * np.pi
   sfs.plot.soundfield(p_point, grid);
   @savefig point_source.png width=50% align=center
   plt.title("Point Source at {} m".format(x0));

Now it looks nice!

For more plotting options, have a look at the documentation of
:func:`sfs.plot.soundfield`.

Now we create a *line source* using the function :func:`sfs.mono.source.line`:

.. ipython:: python

   p_line = sfs.mono.source.line(omega, x0, None, grid)
   p_line *= np.exp(-1j*7*np.pi/4) / np.sqrt(1/(8*np.pi*omega/sfs.defs.c))
   sfs.plot.soundfield(p_line, grid);
   @savefig line_source.png width=50% align=center
   plt.title("Line Source at {} m".format(x0[:2]));

In order to compare the two, let's try to plot them side-by-side.

.. ipython:: python

   f, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
   f.set_figwidth(f.get_figwidth() * 2)
   f.subplots_adjust(wspace=0.05)
   sfs.plot.soundfield(p_point, grid, ax=ax1, colorbar=False);
   ax1.set_title("Point Source");
   im = sfs.plot.soundfield(p_line, grid, ax=ax2, colorbar=False, ylabel="")
   ax2.set_title("Line Source");
   @savefig point_and_line.png width=100%
   f.colorbar(im, ax=[ax1, ax2]);

Finally, let's have a look at a *plane wave*, which can be created with
:func:`sfs.mono.source.plane`.

.. ipython:: python

   plt.close()  # get rid of the double-width figure from above
   direction = 45  # degree
   n0 = sfs.util.normal(np.radians(direction), np.radians(90))
   p_plane = sfs.mono.source.plane(omega, x0, n0, grid)
   sfs.plot.soundfield(p_plane, grid);
   @savefig plane_source.png width=50% align=center
   plt.title("Plane wave with direction {} degree".format(direction));
