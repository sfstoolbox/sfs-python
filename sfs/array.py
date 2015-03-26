"""Compute positions of various secondary source distributions."""

import numpy as np
from . import util


def linear(N, dx, center=[0, 0, 0], n0=[1, 0, 0]):
    """Linear secondary source distribution."""
    center = np.squeeze(np.asarray(center, dtype=np.float64))
    positions = np.zeros((N, 3))
    positions[:, 1] = (np.arange(N) - N / 2 + 1 / 2) * dx
    positions, directions = _rotate_array(positions, [1, 0, 0], [1, 0, 0], n0)
    directions = np.tile(directions, (N, 1))
    positions += center
    weights = dx * np.ones(N)
    return positions, directions, weights


def linear_nested(N, dx1, dx2, center=[0, 0, 0], n0=[1, 0, 0]):
    """Nested linear secondary source distribution."""

    # first segment
    x00, n00, a00 = linear(N//3, dx2, center=[0, -N//6*(dx1+dx2), 0])
    positions = x00
    directions = n00
    # second segment
    x00, n00, a00 = linear(N//3, dx1)
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    # third segment
    x00, n00, a00 = linear(N//3, dx2, center=[0, N//6*(dx1+dx2), 0])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    # compute weights
    weights = weights_linear(positions)
    # shift and rotate array
    positions, directions = _rotate_array(positions, directions, [1, 0, 0], n0)
    positions += center

    return positions, directions, weights


def linear_random(N, dy1, dy2, center=[0, 0, 0], n0=[1, 0, 0]):
    """Randomly sampled linear array."""
    # vector of uniformly distributed random distances between dy2 > dy1
    dist = dy1 + (dy2-dy1)*np.random.rand(N-1)
    # positions of secondary sources
    positions = np.zeros((N, 3))
    for m in range(1, N):
        positions[m, 1] = positions[m-1, 1] + dist[m-1]
    # weights of secondary sources
    weights = weights_linear(positions)
    # directions of scondary sources
    directions = np.tile([1, 0, 0], (N, 1))
    # shift array to center
    positions[:, 1] -= positions[-1, 1] / 2
    # shift and rotate array
    positions, directions = _rotate_array(positions, directions, [1, 0, 0], n0)
    positions += center

    return positions, directions, weights


def circular(N, R, center=[0, 0, 0]):
    """Circular secondary source distribution parallel to the xy-plane."""
    center = np.squeeze(np.asarray(center, dtype=np.float64))
    positions = np.tile(center, (N, 1))
    alpha = np.linspace(0, 2 * np.pi, N, endpoint=False)
    positions[:, 0] += R * np.cos(alpha)
    positions[:, 1] += R * np.sin(alpha)
    directions = np.zeros_like(positions)
    directions[:, 0] = np.cos(alpha + np.pi)
    directions[:, 1] = np.sin(alpha + np.pi)
    weights = R * np.ones(N)
    return positions, directions, weights


def rectangular(Nx, dx, Ny, dy, center=[0, 0, 0], n0=None):
    """Rectangular secondary source distribution."""

    # left array
    x00, n00, a00 = linear(Ny, dy)
    positions = x00
    directions = n00
    weights = a00
    # upper array
    x00, n00, a00 = linear(Nx, dx, center=[Nx/2 * dx, x00[-1, 1] + dy/2, 0],
                           n0=[0, -1, 0])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # right array
    x00, n00, a00 = linear(Ny, dy, center=[x00[-1, 0] + dx/2, 0, 0],
                           n0=[-1, 0, 0])
    x00 = np.flipud(x00)
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # lower array
    x00, n00, a00 = linear(Nx, dx, center=[Nx/2 * dx, x00[-1, 1] - dy/2, 0],
                           n0=[0, 1, 0])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # shift array to center
    positions -= np.asarray([Nx/2 * dx, 0, 0])
    # rotate array
    if n0 is not None:
        positions, directions = _rotate_array(positions, directions,
                                              [1, 0, 0], n0)
    # shift array to desired position
    positions += np.asarray(center)

    return positions, directions, weights


def rounded_edge(Nxy, Nr, dx, center=[0, 0, 0], n0=None):
    """Array along the xy-axis with rounded edge at the origin.

    Parameters
    ----------
    Nxy : integer
        Number of secondary sources along x- and y-axis.
    Nr : integer
        Number of secondary sources in rounded edge. Radius of edge is
        adjusted to equdistant sampling along entire array.
    center : triple of floats
        Position of edge.
    n0 : triple of floats
        Normal vector of array. Default orientation is along xy-axis.

    Returns
    -------
    positions : list of triplets of floats
        positions of secondary sources
    directions : list of triplets of floats
        orientations (normal vectors) of secondary sources
    weights : list of floats
        integration weights of secondary sources

    """

    # radius of rounded edge
    Nr += 1
    R = 2/np.pi * Nr * dx

    # array along y-axis
    x00, n00, a00 = linear(Nxy, dx, center=[0, Nxy//2*dx+dx/2+R, 0])
    x00 = np.flipud(x00)
    positions = x00
    directions = n00
    weights = a00

    # round part
    x00 = np.zeros((Nr, 3))
    n00 = np.zeros((Nr, 3))
    a00 = np.zeros(Nr)
    for n in range(0, Nr):
        alpha = np.pi/2 * n/Nr
        x00[n, 0] = R * (1-np.cos(alpha))
        x00[n, 1] = R * (1-np.sin(alpha))
        n00[n, 0] = np.cos(alpha)
        n00[n, 1] = np.sin(alpha)
        a00[n] = dx
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))

    # array along x-axis
    x00, n00, a00 = linear(Nxy, dx, center=[Nxy//2*dx-dx/2+R, 0, 0],
                           n0=[0, 1, 0])
    x00 = np.flipud(x00)
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))

    # rotate array
    if n0 is not None:
        positions, directions = _rotate_array(positions, directions,
                                              [1, 0, 0], n0)
    # shift array to desired position
    positions += np.asarray(center)

    return positions, directions, weights


def planar(Ny, dy, Nz, dz, center=[0, 0, 0], n0=None):
    """Planar secondary source distribtion."""
    center = np.squeeze(np.asarray(center, dtype=np.float64))
    # initialize vectors for later np.concatenate
    positions = np.zeros((1, 3))
    directions = np.zeros((1, 3))
    for z in (np.arange(Nz) - Nz / 2 + 1 / 2) * dz:
        x00, n00, a00 = linear(Ny, dy, center=[0, 0, z])
        positions = np.concatenate((positions, x00), axis=0)
        directions = np.concatenate((directions, n00), axis=0)
    # remove first element from initialization
    positions = np.delete(positions, 0, axis=0)
    directions = np.delete(directions, 0, axis=0)
    weights = dy * dz * np.ones(Ny*Nz)
    # rotate array
    if n0 is not None:
        positions, directions = _rotate_array(positions, directions,
                                              [1, 0, 0], n0)
    # shift array to desired position
    positions += np.asarray(center)

    return positions, directions, weights


def cube(Nx, dx, Ny, dy, Nz, dz, center=[0, 0, 0], n0=None):
    """Cube shaped secondary source distribtion."""
    center = np.squeeze(np.asarray(center, dtype=np.float64))
    # left array
    x00, n00, a00 = planar(Ny, dy, Nz, dz)
    positions = x00
    directions = n00
    weights = a00
    # upper array
    x00, n00, a00 = planar(Nx, dx, Nz, dz,
                           center=[Nx/2 * dx, x00[-1, 1] + dy/2, 0],
                           n0=[0, -1, 0])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # right array
    x00, n00, a00 = planar(Ny, dy, Nz, dz, center=[x00[-1, 0] + dx/2, 0, 0],
                           n0=[-1, 0, 0])
    x00 = np.flipud(x00)
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # lower array
    x00, n00, a00 = planar(Nx, dx, Nz, dz,
                           center=[Nx/2 * dx, x00[-1, 1] - dy/2, 0],
                           n0=[0, 1, 0])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # bottom array
    x00, n00, a00 = planar(Nx, dx, Ny, dy, center=[Nx/2 * dx, 0, -Nz/2 * dz],
                           n0=[0, 0, 1])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # top array
    x00, n00, a00 = planar(Nx, dx, Ny, dy, center=[Nx/2 * dx, 0, Nz/2 * dz],
                           n0=[0, 0, -1])
    positions = np.concatenate((positions, x00))
    directions = np.concatenate((directions, n00))
    weights = np.concatenate((weights, a00))
    # shift array to center
    positions -= np.asarray([Nx/2 * dx, 0, 0])
    # rotate array
    if n0 is not None:
        positions, directions = _rotate_array(positions, directions,
                                              [1, 0, 0], n0)
    # shift array to desired position
    positions += np.asarray(center)

    return positions, directions, weights


def sphere_load(fname, radius, center=[0, 0, 0]):
    """Spherical secondary source distribution loaded from datafile.

    ASCII Format (see MATLAB SFS Toolbox) with 4 numbers (3 position, 1 weight)
    per secondary source located on the unit circle.
    """
    x0 = np.loadtxt(fname)
    weights = x0[:, 3]
    directions = - x0[:, 0:3]
    positions = center + radius * x0[:, 0:3]

    return positions, directions, weights


def load(fname, center=[0, 0, 0], n0=None):
    """Load secondary source positions from datafile.

       Comma Seperated Values (CSV) format with 7 values
       (3 positions, 3 directions, 1 weight) per secondary source
    """
    data = np.loadtxt(fname, delimiter=',')
    positions = data[:, [0, 1, 2]]
    directions = data[:, [3, 4, 5]]
    weights = np.squeeze(data[:, [6]])
    # rotate array
    if n0 is not None:
        positions, directions = _rotate_array(positions, directions,
                                              [1, 0, 0], n0)
    # shift array to desired position
    positions += np.asarray(center)

    return positions, directions, weights


def weights_linear(positions):
    """Calculate loudspeaker weights for a linear array.

       The linear array has to be parallel to the y-axis.
    """
    N = len(positions)
    weights = np.zeros(N)
    dy = np.diff(positions[:, 1])
    weights[0] = dy[0]
    for m in range(1, N - 1):
        weights[m] = 0.5 * (dy[m-1] + dy[m])
    weights[-1] = dy[-1]

    return np.abs(weights)


def weights_closed(positions):
    """Calculate loudspeaker weights for a simply connected array.

    Note: The loudspeaker positions have to be ordered on the closed contour

    """
    positions = np.asarray(positions)
    if len(positions) == 0:
        weights = []
    elif len(positions) == 1:
        weights = [1.0]
    else:
        successors = np.roll(positions, -1, axis=0)
        d = [np.linalg.norm(b - a) for a, b in zip(positions, successors)]
        weights = [0.5 * (a + b) for a, b in zip(d, d[-1:] + d)]
    return weights


def _rotate_array(x0, n0, n1, n2):
    """Rotate secondary sources from n1 to n2."""
    R = util.rotation_matrix(n1, n2)
    x0 = np.inner(x0, R)
    n0 = np.inner(n0, R)
    return x0, n0
