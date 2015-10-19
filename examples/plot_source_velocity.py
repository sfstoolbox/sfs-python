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
npw = sfs.util.direction_vector(np.radians(pw_angle))

# regular grid for pressure
gridp = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.10)

# random grid for velocity
gridv = [np.random.uniform(-2, 2, 10000), np.random.uniform(-2, 2, 10000), 0]

# compute velocity and pressure field
v = sfs.mono.source.plane_velocity(omega, xs, npw, gridv)
v = sfs.mono.synthesized.shiftphase(v, omega * t)

v = sfs.mono.source.point_velocity(omega, xs, [0,0,0], gridv)

p = sfs.mono.source.plane(omega, xs, npw, gridp)
p = sfs.mono.synthesized.shiftphase(p, omega * t)

# compute displacement
X = gridv + amplitude * sfs.util.displacement(v, omega)

# plot pressure and displacement
fig, ax = plt.subplots(figsize=(10, 10))
sfs.plot.soundfield(p, gridp, ax=ax)
sfs.plot.particles(X, ax=ax, facecolor='black', alpha=.5, s=6)
plt.axis('off')
plt.show()
