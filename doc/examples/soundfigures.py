"""This example illustrates the synthesis of a sound figure.

The sound figure is defined by a grayscale PNG image. Various example
images are located in the "figures/" directory.

"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sfs

dx = 0.10  # secondary source distance
N = 60  # number of secondary sources
pw_angle = [90, 45]  # traveling direction of plane wave
f = 2000  # frequency

# angular frequency
omega = 2 * np.pi * f

# normal vector of plane wave
npw = sfs.util.direction_vector(*np.radians(pw_angle))

# spatial grid
grid = sfs.util.xyz_grid([-3, 3], [-3, 3], 0, spacing=0.02)

# get secondary source positions
array = sfs.array.cube(N, dx)

# driving function for sound figure
figure = np.array(Image.open('figures/tree.png'))  # read image from file
figure = np.rot90(figure)  # turn 0deg to the top
d, selection, secondary_source = sfs.fd.wfs.soundfigure_3d(
    omega, array.x, array.n, figure, npw=npw)

# compute synthesized sound field
p = sfs.fd.synthesize(d, selection, array, secondary_source, grid=grid)

# plot and save synthesized sound field
plt.figure(figsize=(10, 10))
sfs.plot2d.amplitude(p, grid, xnorm=[0, -2.2, 0], cmap='BrBG', colorbar=False,
                     vmin=-1, vmax=1)
plt.title('Synthesized Sound Field')
plt.savefig('soundfigure.png')

# plot and save level of synthesized sound field
plt.figure(figsize=(12.5, 12.5))
im = sfs.plot2d.level(p, grid, xnorm=[0, -2.2, 0], vmin=-50, vmax=0,
                      colorbar_kwargs=dict(label='dB'))
plt.title('Level of Synthesized Sound Field')
plt.savefig('soundfigure_level.png')
