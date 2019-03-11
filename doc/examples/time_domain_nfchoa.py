"""Create some examples of time-domain NFC-HOA."""

import numpy as np
import matplotlib.pyplot as plt
import sfs.time as sfs
from scipy.signal import unit_impulse

# Parameters
fs = 44100  # sampling frequency
grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.005)
N = 60  # number of secondary sources
R = 1.5  # radius of circular array
array = sfs.array.circular(N, R)

# Excitation signal
signal = unit_impulse(512), fs, 0

# Plane wave
max_order = None
npw = [0, -1, 0]  # propagating direction
t = 0  # observation time
delay, weight, sos, phaseshift, selection, secondary_source = \
    sfs.nfchoa.plane_25d(array.x, R, npw, fs, max_order)
d = sfs.nfchoa.driving_signals_25d(
        delay, weight, sos, phaseshift, signal)
p = sfs.synthesize(d, selection, array, secondary_source,
                   observation_time=t, grid=grid)

plt.figure()
sfs.plot.level(p, grid)
sfs.plot.loudspeaker_2d(array.x, array.n)
sfs.plot.virtualsource_2d([0, 0], ns=npw, type='plane')
plt.savefig('impulse_pw_nfchoa_25d.png')

# Point source
max_order = 100
xs = [1.5, 1.5, 0]  # position
t = np.linalg.norm(xs) / sfs.default.c  # observation time
delay, weight, sos, phaseshift, selection, secondary_source = \
    sfs.nfchoa.point_25d(array.x, R, xs, fs, max_order)
d = sfs.nfchoa.driving_signals_25d(
        delay, weight, sos, phaseshift, signal)
p = sfs.synthesize(d, selection, array, secondary_source,
                        observation_time=t, grid=grid)

plt.figure()
sfs.plot.level(p, grid)
sfs.plot.loudspeaker_2d(array.x, array.n)
sfs.plot.virtualsource_2d(xs, type='point')
plt.savefig('impulse_ps_nfchoa_25d.png')
