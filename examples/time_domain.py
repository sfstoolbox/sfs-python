"""
Create some examples in the time domain.

Simulate and plot impulse behavior for Wafe Field Synthesis.

"""

import matplotlib.pyplot as plt
import sfs

grid = sfs.util.xyz_grid([-3, 3], [-3, 3], 0, spacing=0.01)
my_cmap = 'YlOrRd'
N = 56  # number of secondary sources
R = 1.5  # radius of spherical/circular array
x0, nx0, a0 = sfs.array.circular(N, R)  # get secondary source positions
xs = [2, 2, 0]

# compute driving function
d_delay, d_weight, d_line = sfs.time.drivingfunction.wfs_25d_ps(xs, x0, nx0)

# test soundfield
a = sfs.mono.drivingfunction.source_selection_point(nx0, x0, xs)
twin = sfs.tapering.tukey(a, .3)
p = sfs.time.soundfield.synthesize_p(d_line, twin * a0, x0, grid, 200)
p = p * 100  # scale absolute amplitude

plt.figure(figsize=(10, 10))
sfs.plot.level(p, grid, cmap=my_cmap)
sfs.plot.loudspeaker_2d(x0, nx0, twin)
plt.grid()
sfs.plot.virtualsource_2d(xs)
plt.title('impuls_ps_wfs_25d')
plt.savefig('impuls_ps_wfs_25d' + '.png')
