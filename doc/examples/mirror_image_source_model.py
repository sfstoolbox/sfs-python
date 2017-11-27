""" Computes the mirror image sources and the sound field in a rectangular
    room
"""

import numpy as np
import sfs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


L = 2, 2.7, 3  # room dimensions
x0 = 1.2, 1.7, 1.5  # source position
max_order = 2  # maximum order of image sources
coeffs = .8, .8, .6, .6, .7, .7  # wall reflection coefficients
omega = 2*np.pi*1000  # angular frequency of monocromatic sound field
fs = 44100  # sample rate for boadband response
signal = ([1, 0, 0], fs)  # signal for broadband response


# get 2D mirror image sources and their strength
xs, wall_count = sfs.util.image_sources_for_box(x0[0:2], L[0:2], max_order)
source_strength = np.prod(coeffs[0:4]**wall_count, axis=1)
# plot mirror image sources
plt.figure()
plt.scatter(*xs.T, source_strength*20)
plt.gca().add_patch(Rectangle((0, 0), L[0], L[1], fill=False))
plt.xlabel('x / m')
plt.ylabel('y / m')
plt.savefig('image_source_positions.png')


# compute monochromatic sound field
grid = sfs.util.xyz_grid([0, L[0]], [0, L[1]], 1.5, spacing=0.02)
P = sfs.mono.source.point_image_sources(omega, x0, [1, 0, 0], grid, L,
                                        max_order, coeffs=coeffs)
# plot monocromatic sound field
plt.figure()
sfs.plot.soundfield(P, grid, xnorm=[L[0]/2, L[1]/2, L[2]/2])
sfs.plot.virtualsource_2d(x0)
plt.savefig('point_image_sources_mono.png')


# compute spatio-temporal impulse response
grid = sfs.util.xyz_grid([0, L[0]], [0, L[1]], 1.5, spacing=0.005)
p = sfs.time.source.point_image_sources(x0, signal, 0.004, grid, L, max_order,
                                        coeffs=coeffs)
# plot spatio-temporal impulse response
plt.figure()
sfs.plot.level(p, grid)
sfs.plot.virtualsource_2d(x0)
plt.savefig('point_image_sources_time_domain.png')
