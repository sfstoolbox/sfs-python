""" Example for scatter plot visualization of sources """

import numpy as np
import matplotlib.pyplot as plt
import sfs

# simulation parameters
pw_angle = 45  # traveling direction of plane wave
f = 300  # frequency
amplitude = 90000
t = 0.000

# angular frequency
omega = 2 * np.pi * f
# normal vector of plane wave
npw = sfs.util.direction_vector(np.radians(pw_angle), np.radians(90))


# sfs grid object
#grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.5)

# lineary spaced grid
x = np.linspace(-2, 2, num=50)
y = np.linspace(-2, 2, num=50)
grid = sfs.util.asarray_of_arrays( np.meshgrid(x, y) + [0])

# random grid
#x = np.random.uniform(-2, 2, 4000)
#y = np.random.uniform(-2, 2, 4000)
#grid = sfs.util.asarray_of_arrays( [x, y] + [0])

# compute velocity and pressure field
vx, vy, vz = sfs.mono.source.plane_velocity(omega, 0, npw, grid)
p = sfs.mono.source.plane(omega, 0, npw, grid)

# compute and plot displacement
grid = sfs.util.asarray_of_arrays(grid[:-1])

#gridx, gridy = grid[:2]
#grid = sfs.util.asarray_of_arrays([gridx[:2], gridy[:2]])

v = sfs.util.asarray_of_arrays([vx, vy])
v = sfs.mono.synthesized.shiftphase(v, omega*t)
X = grid + amplitude * sfs.util.displacement(v, omega)

fig, ax = plt.subplots(figsize=(10, 10))

sfs.plot.soundfield(p, grid, ax=ax)
sfs.plot.particles(X, ax=ax, c=u'k', alpha=.7)
plt.axis('off')
