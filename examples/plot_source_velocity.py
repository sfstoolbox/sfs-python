""" Example for scatter plot visualization of sources """

import numpy as np
import matplotlib.pyplot as plt
import sfs

# simulation parameters
pw_angle = 45  # traveling direction of plane wave
f = 300  # frequency
amplitude = 70000
t = 0.002

# angular frequency
omega = 2 * np.pi * f
# normal vector of plane wave
npw = sfs.util.direction_vector(np.radians(pw_angle), np.radians(90))


#grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.5)

x = np.linspace(-2, 2, num=50)
y = np.linspace(-2, 2, num=50)
grid = sfs.util.asarray_of_arrays( np.meshgrid(x, y) + [0])

#x = np.random.uniform(-2, 2, 5000)
#y = np.random.uniform(-2, 2, 5000)
#grid = sfs.util.asarray_of_arrays( [x, y] + [0])

#grid = sfs.util.xyz_grid(1, 1, 0, spacing=0.2)

vx, vy, vz = sfs.mono.source.plane_velocity(omega, 0, npw, grid)

# visualization
grid = sfs.util.asarray_of_arrays(grid[:-1])

#gridx, gridy = grid[:2]
#grid = sfs.util.asarray_of_arrays([gridx[:2], gridy[:2]])





v = sfs.util.asarray_of_arrays([vx, vy])

v = sfs.mono.synthesized.shiftphase(v, omega*t)
d = grid + amplitude * sfs.util.displacement(v, omega)

plt.figure(figsize=(10, 10))
sfs.plot.particles(d)
plt.axis('tight')
