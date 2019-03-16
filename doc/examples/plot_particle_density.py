""" Example for particle density visualization of sound sources """

import numpy as np
import matplotlib.pyplot as plt
import sfs

# simulation parameters
pw_angle = 45  # traveling direction of plane wave
xs = [0, 0, 0]  # source position
f = 300  # frequency

# angular frequency
omega = 2 * np.pi * f
# normal vector of plane wave
npw = sfs.util.direction_vector(np.radians(pw_angle))
# random grid for velocity
grid = [np.random.uniform(-3, 3, 40000), np.random.uniform(-3, 3, 40000), 0]


def plot_particle_displacement(title):
    # compute displacement
    X = grid + amplitude * sfs.fd.displacement(v, omega)
    # plot displacement
    plt.figure(figsize=(15, 15))
    plt.cla()
    sfs.plot2d.particles(X, facecolor='black', s=3, trim=[-3, 3, -3, 3])
    plt.axis('off')
    plt.title(title)
    plt.grid()
    plt.savefig(title + '.png')


# point source
v = sfs.fd.source.point_velocity(omega, xs, grid)
amplitude = 1.5e6
plot_particle_displacement('particle_displacement_point_source')

# line source
v = sfs.fd.source.line_velocity(omega, xs, grid)
amplitude = 1.3e6
plot_particle_displacement('particle_displacement_line_source')

# plane wave
v = sfs.fd.source.plane_velocity(omega, xs, npw, grid)
amplitude = 1e5
plot_particle_displacement('particle_displacement_plane_wave')
