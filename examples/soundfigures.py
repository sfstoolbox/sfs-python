"""
  This example illustrates the synthesis of a sound figure.

  The sound figure is defined by a grayscale PNG image. Various examples
  are located in the directory figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sfs


# parameters
dx = 0.10  # secondary source distance
N = 60  # number of secondary sources
pw_angle = [90, 45]  # traveling direction of plane wave
f = 2000  # frequency


# angular frequency
omega = 2 * np.pi * f

# normal vector of plane wave
npw = sfs.util.direction_vector(np.radians(pw_angle[0]), np.radians(pw_angle[1]))

# spatial grid
x = sfs.util.strict_arange(-3, 3, 0.02, endpoint=True)
y = sfs.util.strict_arange(-3, 3, 0.02, endpoint=True)
grid = np.meshgrid(x, y, 0, sparse=True)


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------

# get secondary source positions
x0, n0, a0 = sfs.array.cube(N, dx, N, dx, N, dx)

# driving function for sound figure
figure = np.array(Image.open('figures/tree.png'))  # read image from file
figure = np.rot90(figure)  # turn 0deg to the top
d = sfs.mono.soundfigure.wfs_3d_pw(omega, x0, n0, figure, npw=npw)

# compute synthesized sound field
p = sfs.mono.synthesized.generic(omega, x0, n0, d * a0, grid,
                                 source=sfs.mono.source.point)

# plot and save synthesized sound field
plt.figure(figsize=(10, 10))
sfs.plot.soundfield(2.5e-9 * p, grid, colorbar=False, cmap=plt.cm.BrBG)
plt.title('Synthesized Sound Field')
plt.savefig('soundfigure.png')


# plot and save level of synthesized sound field
plt.figure(figsize=(12.5, 12.5))
Lp = 20*np.log10(abs(p) / abs(p[len(x)//2, len(y)//2]))
plt.imshow(Lp, origin='lower',
           extent=[min(x), max(x), min(y), max(y)],
           vmin=-50, vmax=0, aspect='equal')
plt.title('Level of Synthesized Sound Field')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
cbar = plt.colorbar(shrink=0.8)
cbar.set_label('dB')
plt.savefig('soundfigure_level.png')
