""" Example for scatter plot visualization of sources """

import numpy as np
import matplotlib.pyplot as plt
import sfs

# simulation parameters
pw_angle = 45  # traveling direction of plane wave
f = 300  # frequency
amplitude = 100000  # amplification of velocity
t = 0.000  # time
xs = [0, 0, 0]  # source position

# angular frequency
omega = 2 * np.pi * f
# normal vector of plane wave
npw = sfs.util.direction_vector(np.radians(pw_angle), np.radians(90))


# sfs grid object for pressure
gridp = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.10)

# lineary spaced grid for velocity
#x = np.linspace(-2, 2, num=50)
#y = np.linspace(-2, 2, num=50)
#gridv = sfs.util.asarray_of_arrays( np.meshgrid(x, y) + [0])

# random grid for velocity
x = np.random.uniform(-2, 2, 10000)
y = np.random.uniform(-2, 2, 10000)
gridv = sfs.util.asarray_of_arrays( [x, y] + [0])

# compute velocity and pressure field
vx, vy, vz = sfs.mono.source.plane_velocity(omega, xs, npw, gridv)
#vx, vy, vz = sfs.mono.source.point_velocity(omega, xs, npw, gridv)

p = sfs.mono.source.plane(omega, 0, npw, gridp)

# compute and plot displacement
gridv = sfs.util.asarray_of_arrays(gridv[:-1])

v = sfs.util.asarray_of_arrays([vx, vy])
v = sfs.mono.synthesized.shiftphase(v, omega*t)
X = gridv + amplitude * sfs.util.displacement(v, omega)


# plot pressure and displacement
fig, ax = plt.subplots(figsize=(10, 10))

sfs.plot.soundfield(p, gridp, ax=ax)
sfs.plot.particles(X, ax=ax, c=u'k', alpha=.5, linewidths=0, s=6)
plt.axis('off')
