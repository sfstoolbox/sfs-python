# example thats illustrates the application of python to the numeric simulation
# of SFS

import matplotlib.pyplot as plt
import numpy as np
import sfs


# parameters
dx = 0.1  # secondary source distance
N = 50  # number of secondary sources
pw_angle = np.pi / 4  # traveling direction of plane wave
f = 1000  # frequency


# wavenumber
k = 2 * np.pi * f / 343

# normal vector of plane wave
npw = np.array([np.cos(pw_angle), np.sin(pw_angle), 0])

# spatial grid
x = np.arange(-2, 2, 0.02)
y = np.arange(-2, 2, 0.02)


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------

# get secondary source positions
#x0,n0 = sfs.array.linear(N, dx)
x0, n0 = sfs.array.circular(N, 1)

# get driving function
#d = sfs.drivingfunction.delay_3d_pw(k, x0, n0, npw)
d = sfs.drivingfunction.wfs_3d_pw(k, x0, n0, npw)

# get active secondary sources
a = sfs.drivingfunction.source_selection_pw(n0, npw)

# get tapering window
#twin = sfs.tapering.none(a)
twin = sfs.tapering.kaiser(a)

# compute synthesized sound field
p = sfs.synthesized.generic(x, y, 0, x0, k, d, twin)

# plot synthesized sound field
plt.figure(figsize=(15, 15))

plt.imshow(np.real(p), cmap=plt.cm.RdBu, origin='lower',
           extent=[-2, 2, -2, 2], vmax=2, vmin=-2, aspect='equal')
plt.colorbar()

plt.savefig('soundfield.png')