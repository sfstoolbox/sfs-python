"""
Create some examples in the time domain.

Simulate and plot impulse behavior for Wave Field Synthesis.

"""

import numpy as np
import matplotlib.pyplot as plt
import sfs

# simulation parameters
grid = sfs.util.xyz_grid([-3, 3], [-3, 3], 0, spacing=0.01)
my_cmap = 'YlOrRd'
N = 56  # number of secondary sources
R = 1.5  # radius of spherical/circular array
x0, n0, a0 = sfs.array.circular(N, R)  # get secondary source positions
fs = 44100  # sampling rate

# unit impulse
signal = [1], fs

# POINT SOURCE
xs = [2, 2, 0]  # position of virtual source
t = 0.008
# compute driving signals
d_delay, d_weight = sfs.time.drivingfunction.wfs_25d_point(x0, n0, xs)
d = sfs.time.drivingfunction.driving_signals(d_delay, d_weight, signal)

# test soundfield
a = sfs.mono.drivingfunction.source_selection_point(n0, x0, xs)
twin = sfs.tapering.tukey(a, .3)
p = sfs.time.soundfield.p_array(x0, d, twin * a0, t, grid)
p = p * 100  # scale absolute amplitude

plt.figure(figsize=(10, 10))
sfs.plot.level(p, grid, cmap=my_cmap)
sfs.plot.loudspeaker_2d(x0, n0, twin)
plt.grid()
sfs.plot.virtualsource_2d(xs)
plt.title('impulse_ps_wfs_25d')
plt.savefig('impulse_ps_wfs_25d.png')

# PLANE WAVE
pw_angle = 30  # traveling direction of plane wave
npw = sfs.util.direction_vector(np.radians(pw_angle))
t = -0.001

# compute driving signals
d_delay, d_weight = sfs.time.drivingfunction.wfs_25d_plane(x0, n0, npw)
d = sfs.time.drivingfunction.driving_signals(d_delay, d_weight, signal)

# test soundfield
a = sfs.mono.drivingfunction.source_selection_plane(n0, npw)
twin = sfs.tapering.tukey(a, .3)
p = sfs.time.soundfield.p_array(x0, d, twin * a0, t, grid)

plt.figure(figsize=(10, 10))
sfs.plot.level(p, grid, cmap=my_cmap)
sfs.plot.loudspeaker_2d(x0, n0, twin)
plt.grid()
sfs.plot.virtualsource_2d([0, 0], npw, type='plane')
plt.title('impulse_pw_wfs_25d')
plt.savefig('impulse_pw_wfs_25d.png')

# FOCUSED SOURCE
xs = np.r_[0.5, 0.5, 0]  # position of virtual source
xref = np.r_[0, 0, 0]
nfs = sfs.util.normalize_vector(xref - xs)  # main n of fsource
t = 0.003  # compute driving signals
d_delay, d_weight = sfs.time.drivingfunction.wfs_25d_focused(x0, n0, xs)
d = sfs.time.drivingfunction.driving_signals(d_delay, d_weight, signal)

# test soundfield
a = sfs.mono.drivingfunction.source_selection_focused(nfs, x0, xs)
twin = sfs.tapering.tukey(a, .3)
p = sfs.time.soundfield.p_array(x0, d, twin * a0, t, grid)
p = p * 100  # scale absolute amplitude

plt.figure(figsize=(10, 10))
sfs.plot.level(p, grid, cmap=my_cmap)
sfs.plot.loudspeaker_2d(x0, n0, twin)
plt.grid()
sfs.plot.virtualsource_2d(xs)
plt.title('impulse_fs_wfs_25d')
plt.savefig('impulse_fs_wfs_25d.png')

# plt.show()
