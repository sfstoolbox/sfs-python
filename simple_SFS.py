# example thats illustrates the application of python to the numeric simulation
# of SFS
#
# Sascha Spors, Sascha.Spors@uni-rostock.de

import matplotlib.pyplot as plt
import math
import numpy as np
import sfs


# parameters
dx = 0.1  # secondary source distance
N = 50  # number of secondary sources
pw_angle = math.pi/4  # traveling direction of plane wave
f = 1000  # frequency



# wavenumber
k = 2 * math.pi * f / 343

# normal vector of plane wave
npw = np.array([np.cos(pw_angle), np.sin(pw_angle), 0])

# spatial grid
x = np.arange(-5, 5, 0.02)
y = np.arange(0, 5, 0.02)


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------

# get secondary source positions
x0 = sfs.array.linear(N, dx)

# get driving function
d = sfs.drivingfunction.pw_delay(k, x0, npw)

# get tapering window
twin = sfs.tapering.weight(N)

# compute synthesized sound field
p = sfs.synthesized.generic(x, y, x0, k, d, twin)


# plot synthesized sound field
plt.figure(figsize = (15, 15))

plt.imshow(np.real(p), cmap=plt.cm.RdBu, origin='lower', extent=[-5,5,0,5], vmax=10, vmin=-10, aspect='equal')
plt.colorbar()

plt.savefig('soundfield.png')
