"""Compute driving functions for sound figures."""

import numpy as np
from .. import util
from . import drivingfunction


def wfs_3d_pw(omega, x0, n0, figure, npw=[0, 0, 1], c=None):
    """Compute driving function for a 2D sound figure.

    Based on
    [Helwani et al., The Synthesis of Sound Figures, MSSP, 2013]

    """

    x0 = np.asarray(x0)
    n0 = np.asarray(n0)
    k = util.wavenumber(omega, c)
    nx, ny = figure.shape

    # 2D spatial DFT of image
    figure = np.fft.fftshift(figure, axes=(0, 1))  # sign of spatial DFT
    figure = np.fft.fft2(figure)
    # wavenumbers
    kx = np.fft.fftfreq(nx, 1./nx)
    ky = np.fft.fftfreq(ny, 1./ny)
    # shift spectrum due to desired plane wave
    figure = np.roll(figure, int(k*npw[0]), axis=0)
    figure = np.roll(figure, int(k*npw[1]), axis=1)
    # search and iterate over propagating plane wave components
    kxx, kyy = np.meshgrid(kx, ky, sparse=True)
    rho = np.sqrt((kxx) ** 2 + (kyy) ** 2)
    d = 0
    for n in range(nx):
        for m in range(ny):
            if(rho[n, m] < k):
                # dispertion relation
                kz = np.sqrt(k**2 - rho[n, m]**2)
                # normal vector of plane wave
                npw = 1/k * np.asarray([kx[n], ky[m], kz])
                npw = npw / np.linalg.norm(npw)
                # driving function of plane wave with positive kz
                a = drivingfunction.source_selection_plane(n0, npw)
                a = a * figure[n, m]
                d += a * drivingfunction.wfs_3d_plane(omega, x0, n0, npw, c)

    return d
