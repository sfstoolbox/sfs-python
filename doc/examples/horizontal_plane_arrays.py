"""
    Generates sound fields for various arrays and virtual source types.
"""

import numpy as np
import matplotlib.pyplot as plt
import sfs


dx = 0.1  # secondary source distance
N = 30  # number of secondary sources
f = 1000  # frequency
pw_angle = 20  # traveling direction of plane wave
xs = [-1.5, 0.2, 0]  # position of virtual monopole
tapering = sfs.tapering.tukey  # tapering window
talpha = 0.3  # parameter for tapering window
xnorm = [1, 1, 0]  # normalization point for plots
grid = sfs.util.xyz_grid([-2.5, 2.5], [-1.5, 2.5], 0, spacing=0.02)
acenter = [0.3, 0.7, 0]  # center and normal vector of array
anormal = sfs.util.direction_vector(np.radians(35), np.radians(90))

# angular frequency
omega = 2 * np.pi * f
# normal vector of plane wave
npw = sfs.util.direction_vector(np.radians(pw_angle), np.radians(90))


def compute_and_plot_soundfield(title):
    """Compute and plot synthesized sound field."""
    print('Computing', title)

    twin = tapering(selection, alpha=talpha)
    p = sfs.fd.synthesize(d, twin, array, secondary_source, grid=grid)

    plt.figure(figsize=(15, 15))
    plt.cla()
    sfs.plot2d.amplitude(p, grid, xnorm=xnorm)
    sfs.plot2d.loudspeakers(array.x, array.n, twin)
    sfs.plot2d.virtualsource(xs)
    sfs.plot2d.virtualsource([0, 0], npw, type='plane')
    plt.title(title)
    plt.grid()
    plt.savefig(title + '.png')


# linear array, secondary point sources, virtual monopole
array = sfs.array.linear(N, dx, center=acenter, orientation=anormal)

d, selection, secondary_source = sfs.fd.wfs.point_3d(
    omega, array.x, array.n, xs)
compute_and_plot_soundfield('linear_ps_wfs_3d_point')

d, selection, secondary_source = sfs.fd.wfs.point_25d(
    omega, array.x, array.n, xs, xref=xnorm)
compute_and_plot_soundfield('linear_ps_wfs_25d_point')

d, selection, secondary_source = sfs.fd.wfs.point_2d(
    omega, array.x, array.n, xs)
compute_and_plot_soundfield('linear_ps_wfs_2d_point')

# linear array, secondary line sources, virtual line source
d, selection, secondary_source = sfs.fd.wfs.line_2d(
    omega, array.x, array.n, xs)
compute_and_plot_soundfield('linear_ls_wfs_2d_line')


# linear array, secondary point sources, virtual plane wave
d, selection, secondary_source = sfs.fd.wfs.plane_3d(
    omega, array.x, array.n, npw)
compute_and_plot_soundfield('linear_ps_wfs_3d_plane')

d, selection, secondary_source = sfs.fd.wfs.plane_25d(
    omega, array.x, array.n, npw, xref=xnorm)
compute_and_plot_soundfield('linear_ps_wfs_25d_plane')

d, selection, secondary_source = sfs.fd.wfs.plane_2d(
    omega, array.x, array.n, npw)
compute_and_plot_soundfield('linear_ps_wfs_2d_plane')


# non-uniform linear array, secondary point sources
array = sfs.array.linear_diff(N//3 * [dx] + N//3 * [dx/2] + N//3 * [dx],
                                   center=acenter, orientation=anormal)

d, selection, secondary_source = sfs.fd.wfs.point_25d(
    omega, array.x, array.n, xs, xref=xnorm)
compute_and_plot_soundfield('linear_nested_ps_wfs_25d_point')

d, selection, secondary_source = sfs.fd.wfs.plane_25d(
    omega, array.x, array.n, npw, xref=xnorm)
compute_and_plot_soundfield('linear_nested_ps_wfs_25d_plane')


# random sampled linear array, secondary point sources
array = sfs.array.linear_random(N, dx/2, 1.5*dx, center=acenter,
                                     orientation=anormal)

d, selection, secondary_source = sfs.fd.wfs.point_25d(
    omega, array.x, array.n, xs, xref=xnorm)
compute_and_plot_soundfield('linear_random_ps_wfs_25d_point')

d, selection, secondary_source = sfs.fd.wfs.plane_25d(
    omega, array.x, array.n, npw, xref=xnorm)
compute_and_plot_soundfield('linear_random_ps_wfs_25d_plane')


# rectangular array, secondary point sources
array = sfs.array.rectangular((N, N//2), dx, center=acenter, orientation=anormal)
d, selection, secondary_source = sfs.fd.wfs.point_25d(
    omega, array.x, array.n, xs, xref=xnorm)
compute_and_plot_soundfield('rectangular_ps_wfs_25d_point')

d, selection, secondary_source = sfs.fd.wfs.plane_25d(
    omega, array.x, array.n, npw, xref=xnorm)
compute_and_plot_soundfield('rectangular_ps_wfs_25d_plane')


# circular array, secondary point sources
N = 60
array = sfs.array.circular(N, 1, center=acenter)
d, selection, secondary_source = sfs.fd.wfs.point_25d(
    omega, array.x, array.n, xs, xref=xnorm)
compute_and_plot_soundfield('circular_ps_wfs_25d_point')

d, selection, secondary_source = sfs.fd.wfs.plane_25d(
    omega, array.x, array.n, npw, xref=xnorm)
compute_and_plot_soundfield('circular_ps_wfs_25d_plane')


# circular array, secondary line sources, NFC-HOA
array = sfs.array.circular(N, 1)
xnorm = [0, 0, 0]
talpha = 0  # switches off tapering

d, selection, secondary_source = sfs.fd.nfchoa.plane_2d(
    omega, array.x, 1, npw)
compute_and_plot_soundfield('circular_ls_nfchoa_2d_plane')


# circular array, secondary point sources, NFC-HOA
array = sfs.array.circular(N, 1)
xnorm = [0, 0, 0]
talpha = 0  # switches off tapering

d, selection, secondary_source = sfs.fd.nfchoa.point_25d(
    omega, array.x, 1, xs)
compute_and_plot_soundfield('circular_ps_nfchoa_25d_point')

d, selection, secondary_source = sfs.fd.nfchoa.plane_25d(
    omega, array.x, 1, npw)
compute_and_plot_soundfield('circular_ps_nfchoa_25d_plane')
