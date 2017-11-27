"""
  This example illustrates the use of the modal room model.

"""

import numpy as np
import matplotlib.pyplot as plt
import sfs

x0 = [1, 3, 1.80]  # source position
L = [6, 6, 3]  # dimensions of room
deltan = 0.01  # absorption factor of walls
n0 = [1, 0, 0]  # normal vector of source (only for compatibilty)
N = 20  # maximum order of modes
#N = [1, 0, 0]  # room mode to compute

fresponse = True  # freqeuency response or sound field?

# compute and plot frequency response at one point
if fresponse:
    f = np.linspace(20, 200, 180)  # frequency
    omega = 2 * np.pi * f  # angular frequency
    grid = sfs.util.xyz_grid(1, 1, 1.80, spacing=1)

    p = []
    for om in omega:
        p.append(sfs.mono.source.point_modal(om, x0, n0, grid, L,
                                             N=N, deltan=deltan))

    p = np.asarray(p)

    plt.plot(f, 20*np.log10(np.abs(p)))
    plt.xlabel('frequency / Hz')
    plt.ylabel('level / dB')
    plt.grid()


# compute and plot sound field for one frequency
if not fresponse:
    f = 500  # frequency
    omega = 2 * np.pi * f  # angular frequency
    grid = sfs.util.xyz_grid([0, L[0]], [0, L[1]], L[2], spacing=.1)

    p = sfs.mono.source.point_modal(omega, x0, n0, grid, L, N=N, deltan=deltan)

    sfs.plot.soundfield(p, grid, xnorm=[2, 3, 0], colorbar=False,
                        vmax=2, vmin=-2)
