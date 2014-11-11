# example thats illustrates the application of python to the numeric simulation
# of SFS

import matplotlib.pyplot as plt
import numpy as np
import sfs


# parameters
dx = 0.1  # secondary source distance
N = 50  # number of secondary sources
pw_angle = np.radians(90)  # traveling direction of plane wave
xs = [0, 2, 0]  # position of virtual source
f = 1000  # frequency


# angular frequency
omega = 2 * np.pi * f

# normal vector of plane wave
npw = np.array([np.cos(pw_angle), np.sin(pw_angle), 0])

# spatial grid
x = np.arange(-2, 2, 0.01)
y = np.arange(-2, 2, 0.01)


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------

# get secondary source positions
#x0,n0 = sfs.array.linear(N, dx)
x0, n0 = sfs.array.circular(N, 1)

# get driving function
#d = sfs.mono.drivingfunction.delay_3d_pw(omega, x0, n0, npw)

#d = sfs.mono.drivingfunction.wfs_2d_pw(omega, x0, n0, npw)
d = sfs.mono.drivingfunction.wfs_25d_pw(omega, x0, n0, npw)
#d = sfs.mono.drivingfunction.wfs_3d_pw(omega, x0, n0, npw)

#d = sfs.mono.drivingfunction.wfs_2d_ps(omega, x0, n0, xs)
#d = sfs.mono.drivingfunction.wfs_25d_ps(omega, x0, n0, xs)
#d = sfs.mono.drivingfunction.wfs_3d_ps(omega, x0, n0, xs)


# get active secondary sources
a = sfs.mono.drivingfunction.source_selection_pw(n0, npw)
#a = sfs.mono.drivingfunction.source_selection_ps(n0, x0, xs)

# get tapering window
#twin = sfs.tapering.none(a)
twin = sfs.tapering.kaiser(a)

# compute synthesized sound field
p = sfs.mono.synthesized.generic(omega, x0, d * twin, x, y, 0)


# plot synthesized sound field
plt.figure(figsize=(15, 15))

plt.imshow(np.real(p), cmap=plt.cm.RdBu, origin='lower',
           extent=[-2, 2, -2, 2], vmax=100, vmin=-100, aspect='equal')
plt.colorbar()

sfs.plot.loudspeaker(x0, n0)

plt.savefig('soundfield.png')