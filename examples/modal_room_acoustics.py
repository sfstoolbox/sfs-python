import numpy as np
import matplotlib.pyplot as plt
import sfs

x0 = [1, 3, 1.80]  # source position
L = [6, 6, 3]  # dimensions of room
N = 20  # number of modal components per dimension
deltan = 0.1  # absorption factor of walls


# compute frequency response
if False is True:
    f = np.linspace(20, 200, 180)  # frequency
    omega = 2 * np.pi * f  # angular frequency
    grid = sfs.util.xyz_grid(1, 1, 1.80, spacing=1)

    p = []
    for om in omega:
        p.append(sfs.mono.source.point_modal(om, x0, grid, L, N, deltan))

    p = np.asarray(p)

    plt.plot(f, 20*np.log10(np.abs(p)))
    plt.grid()


# compute sound field
if True is True:
    f = 500  # frequency
    omega = 2 * np.pi * f  # angular frequency
    grid = sfs.util.xyz_grid([0, 6], [0, 6], 1.80, spacing=.1)

    p = sfs.mono.source.point_modal(omega, x0, grid, L, N=[2, 0, 0], deltan=deltan)

    sfs.plot.soundfield(p, grid, xnorm=[3, 3, 0], colorbar=False, vmax=1.5, vmin=-1.5)