"""Generates sound fields for all implemented arrays and
virtual source types."""

import numpy as np
import matplotlib.pyplot as plt
import sfs


# parameters
dx = 0.1  # secondary source distance
N = 30  # number of secondary sources
pw_angle = 20  # traveling direction of plane wave
xs = [-1, 0.1, 0]  # position of virtual source
f = 1000  # frequency
tapering = sfs.tapering.kaiser

# angular frequency
omega = 2 * np.pi * f
# normal vector of plane wave
npw = sfs.util.normal(np.radians(pw_angle), np.radians(90))
# spatial grid
x = np.arange(-2.5, 2.5, 0.02)
y = np.arange(-2.5, 2.5, 0.02)



def plot_soundfield(p, x, y, xnorm, title):
    """plot synthesized sound field."""
    plt.figure(figsize=(15, 15))
    sfs.plot.soundfield(p, x, y, [1, 0, 0])
    sfs.plot.loudspeaker_2d(x0, n0, twin)
    plt.title(title)
    plt.grid()
    plt.savefig(title + '.png')



# linear array, secondary point sources, virtual monopole
x0, n0, a0 = sfs.array.linear(N, dx, center=[0.3, 0.7, 0],
                              n0=sfs.util.normal(np.radians(35), np.radians(90)))
a = sfs.mono.drivingfunction.source_selection_point(n0, x0, xs)
twin = tapering(a)
xnorm = [1, 1, 0]

d = sfs.mono.drivingfunction.wfs_3d_point(omega, x0, n0, xs)
p = sfs.mono.synthesized.generic(omega, x0, d * twin * a0 , x, y, 0,
                                 source=sfs.mono.source.point)
plot_soundfield(p, x, y, xnorm, 'linear_ps_wfs_3d_point')

d = sfs.mono.drivingfunction.wfs_25d_point(omega, x0, n0, xs, xref=xnorm)
p = sfs.mono.synthesized.generic(omega, x0, d * twin * a0 , x, y, 0,
                                 source=sfs.mono.source.point)
plot_soundfield(p, x, y, xnorm, 'linear_ps_wfs_25d_point')

d = sfs.mono.drivingfunction.wfs_2d_point(omega, x0, n0, xs)
p = sfs.mono.synthesized.generic(omega, x0, d * twin * a0 , x, y, 0,
                                 source=sfs.mono.source.point)
plot_soundfield(p, x, y, xnorm, 'linear_ps_wfs_2d_point')


# linear array, secondary point sources, virtual plane wave
d = sfs.mono.drivingfunction.wfs_3d_plane(omega, x0, n0, npw)
p = sfs.mono.synthesized.generic(omega, x0, d * twin * a0 , x, y, 0,
                                 source=sfs.mono.source.point)
plot_soundfield(p, x, y, xnorm, 'linear_ps_wfs_3d_plane')

d = sfs.mono.drivingfunction.wfs_25d_plane(omega, x0, n0, npw, xref=xnorm)
p = sfs.mono.synthesized.generic(omega, x0, d * twin * a0 , x, y, 0,
                                 source=sfs.mono.source.point)
plot_soundfield(p, x, y, xnorm, 'linear_ps_wfs_25d_plane')

d = sfs.mono.drivingfunction.wfs_2d_plane(omega, x0, n0, npw)
p = sfs.mono.synthesized.generic(omega, x0, d * twin * a0 , x, y, 0,
                                 source=sfs.mono.source.point)
plot_soundfield(p, x, y, xnorm, 'linear_ps_wfs_2d_plane')


# linear array, secondary line sources, virtual line source
d = sfs.mono.drivingfunction.wfs_2d_line(omega, x0, n0, xs)
p = sfs.mono.synthesized.generic(omega, x0, d * twin * a0 , x, y, 0,
                                 source=sfs.mono.source.line)
plot_soundfield(p, x, y, xnorm, 'linear_ls_wfs_2d_line')


# non-uniform linear array, secondary point sources, virtual monopole
x0, n0, a0 = sfs.array.linear_nonuniform(N, dx/2, dx, center=[0.3, 0.7, 0],
                              n0=sfs.util.normal(np.radians(35), np.radians(90)))
a = sfs.mono.drivingfunction.source_selection_point(n0, x0, xs)
twin = tapering(a)
xnorm = [1, 1, 0]

d = sfs.mono.drivingfunction.wfs_25d_point(omega, x0, n0, xs, xref=xnorm)
p = sfs.mono.synthesized.generic(omega, x0, d * twin * a0 , x, y, 0,
                                 source=sfs.mono.source.point)
plot_soundfield(p, x, y, xnorm, 'linear_nonuniform_ps_wfs_25d_point')

d = sfs.mono.drivingfunction.wfs_25d_plane(omega, x0, n0, npw, xref=xnorm)
p = sfs.mono.synthesized.generic(omega, x0, d * twin * a0 , x, y, 0,
                                 source=sfs.mono.source.point)
plot_soundfield(p, x, y, xnorm, 'linear_nonuniform_ps_wfs_25d_plane')


# random sampled linear array, secondary point sources, virtual monopole
x0, n0, a0 = sfs.array.linear_random(N, dx/2, 1.5*dx, center=[0.3, 0.7, 0],
                              n0=sfs.util.normal(np.radians(35), np.radians(90)))
a = sfs.mono.drivingfunction.source_selection_point(n0, x0, xs)
twin = tapering(a)
xnorm = [1, 1, 0]

d = sfs.mono.drivingfunction.wfs_25d_point(omega, x0, n0, xs, xref=xnorm)
p = sfs.mono.synthesized.generic(omega, x0, d * twin * a0 , x, y, 0,
                                 source=sfs.mono.source.point)
plot_soundfield(p, x, y, xnorm, 'linear_random_ps_wfs_25d_point')

d = sfs.mono.drivingfunction.wfs_25d_plane(omega, x0, n0, npw, xref=xnorm)
p = sfs.mono.synthesized.generic(omega, x0, d * twin * a0 , x, y, 0,
                                 source=sfs.mono.source.point)
plot_soundfield(p, x, y, xnorm, 'linear_random_ps_wfs_25d_plane')
