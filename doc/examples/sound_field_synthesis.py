"""
    Illustrates the usage of the SFS toolbox for the simulation of SFS.

    This script contains almost all possibilities than can be used
    for the synthesis of sound fields generated by Wave Field Synthesis or
    Higher-Order Ambisonics with various loudspeaker configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sfs.frequency as sfs


# simulation parameters
dx = 0.2  # secondary source distance
N = 16  # number of secondary sources
pw_angle = 30  # traveling direction of plane wave
xs = [2, 1, 0]  # position of virtual source
xref = [0, 0, 0]  # reference position for 2.5D
f = 680  # frequency
R = 1.5  # radius of spherical/circular array

grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.02)

# angular frequency
omega = 2 * np.pi * f
# normal vector of plane wave
npw = sfs.util.direction_vector(np.radians(pw_angle))


# === get secondary source positions ===
#array = sfs.array.linear(N, dx, center=[-1, 0, 0])
#array = sfs.array.linear_random(N, 0.2*dx, 5*dx)
#array = sfs.array.rectangular(N, dx, orientation=sfs.util.direction_vector(0*np.pi/4))
array = sfs.array.circular(N, R)
#array = sfs.array.load('../../data/arrays/wfs_university_rostock_2018.csv')
#array.x[:,2] = 0  # in wfs_university_rostock_2018.csv we encode absolute height
# which is not used here, we also could set the grid coordinate to z=1.615 m

#array = sfs.array.planar(N, dx, orientation=sfs.util.direction_vector(np.radians(0), np.radians(180)))
#array = sfs.array.cube(N, dx, orientation=sfs.util.direction_vector(0, np.pi/2))

#array = sfs.array.sphere_load('/Users/spors/Documents/src/SFS/data/spherical_grids/equally_spaced_points/006561points.mat', 1, center=[.5,0,0])


# === compute driving function and determine active secondary sources ===
#d, selection, secondary_source = sfs.wfs.plane_3d_delay(omega, array.x, array.n, npw)

#d, selection, secondary_source = sfs.wfs.line_2d(omega, array.x, array.n, xs)

#d, selection, secondary_source = sfs.wfs.plane_2d(omega, array.x, array.n, npw)
d, selection, secondary_source = sfs.wfs.plane_25d(omega, array.x, array.n, npw, xref)
#d, selection, secondary_source = sfs.wfs.plane_3d(omega, array.x, array.n, npw)

#d, selection, secondary_source = sfs.wfs.point_2d(omega, array.x, array.n, xs)
#d, selection, secondary_source = sfs.wfs.point_25d(omega, array.x, array.n, xs)
#d, selection, secondary_source = sfs.wfs.point_3d(omega, array.x, array.n, xs)

#d, selection, secondary_source = sfs.nfchoa.plane_2d(omega, array.x, R, npw)

#d, selection, secondary_source = sfs.nfchoa.point_25d(omega, array.x, R, xs)
#d, selection, secondary_source = sfs.nfchoa.plane_25d(omega, array.x, R, npw)


# === compute tapering window ===
#twin = sfs.tapering.none(selection)
#twin = sfs.tapering.kaiser(selection, 8.6)
twin = sfs.tapering.tukey(selection, 0.3)

# === compute synthesized sound field ===
p = sfs.synthesize(d, twin, array, secondary_source, grid=grid)


# === plot synthesized sound field ===
plt.figure(figsize=(10, 10))
sfs.plot.soundfield(p, grid, [0, 0, 0])
sfs.plot.loudspeaker_2d(array.x, array.n, twin)
plt.grid()
plt.savefig('soundfield.png')


#sfs.plot.loudspeaker_3d(array.x, array.n, twin)
#plt.savefig('loudspeakers.png')
