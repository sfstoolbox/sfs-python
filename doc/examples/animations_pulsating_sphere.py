"""Animations of pulsating sphere."""
import sfs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


def particle_displacement(omega, center, radius, amplitude, grid, frames,
                          figsize=(8, 8), interval=80, blit=True, **kwargs):
    """Generate sound particle animation."""
    velocity = sfs.fd.source.pulsating_sphere_velocity(
               omega, center, radius, amplitude, grid)
    displacement = sfs.fd.displacement(velocity, omega)
    phasor = np.exp(1j * 2 * np.pi / frames)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis([grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()])
    scat = sfs.plot2d.particles(grid + displacement, **kwargs)

    def update_frame_displacement(i):
        position = (grid + displacement * phasor**i).apply(np.real)
        position = np.column_stack([position[0].flatten(),
                                    position[1].flatten()])
        scat.set_offsets(position)
        return [scat]

    return animation.FuncAnimation(
            fig, update_frame_displacement, frames,
            interval=interval, blit=blit)


def particle_velocity(omega, center, radius, amplitude, grid, frames,
                      figsize=(8, 8), interval=80, blit=True, **kwargs):
    """Generate particle velocity animation."""
    velocity = sfs.fd.source.pulsating_sphere_velocity(
               omega, center, radius, amplitude, grid)
    phasor = np.exp(1j * 2 * np.pi / frames)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis([grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()])
    quiv = sfs.plot2d.vectors(
            velocity, grid, clim=[-omega * amplitude, omega * amplitude],
            **kwargs)

    def update_frame_velocity(i):
        quiv.set_UVC(*(velocity[:2] * phasor**i).apply(np.real))
        return [quiv]

    return animation.FuncAnimation(
            fig, update_frame_velocity, frames, interval=interval, blit=True)


def sound_pressure(omega, center, radius, amplitude, grid, frames,
                   pulsate=False, figsize=(8, 8), interval=80, blit=True,
                   **kwargs):
    """Generate sound pressure animation."""
    pressure = sfs.fd.source.pulsating_sphere(
            omega, center, radius, amplitude, grid, inside=pulsate)
    phasor = np.exp(1j * 2 * np.pi / frames)

    fig, ax = plt.subplots(figsize=figsize)
    im = sfs.plot2d.amplitude(np.real(pressure), grid, **kwargs)
    ax.axis([grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()])

    def update_frame_pressure(i):
        distance = np.linalg.norm(grid)
        p = pressure * phasor**i
        if pulsate:
            p[distance <= radius + amplitude * np.real(phasor**i)] = np.nan
        im.set_array(np.real(p))
        return [im]

    return animation.FuncAnimation(
            fig, update_frame_pressure, frames, interval=interval, blit=True)


if __name__ == '__main__':

    # Pulsating sphere
    center = [0, 0, 0]
    radius = 0.25
    f = 750  # frequency
    omega = 2 * np.pi * f  # angular frequency

    # Axis limits
    xmin, xmax = -1, 1
    ymin, ymax = -1, 1

    # Animations
    frames = 20  # frames per period

    # Particle displacement
    amplitude = 5e-2  # amplitude of the surface displacement
    grid = sfs.util.xyz_grid([xmin, xmax], [ymin, ymax], 0, spacing=0.025)
    ani = particle_displacement(
            omega, center, radius, amplitude, grid, frames, c='Gray')
    ani.save('pulsating_sphere_displacement.gif', dpi=80, writer='imagemagick')

    # Particle velocity
    amplitude = 1e-3  # amplitude of the surface displacement
    grid = sfs.util.xyz_grid([xmin, xmax], [ymin, ymax], 0, spacing=0.04)
    ani = particle_velocity(
            omega, center, radius, amplitude, grid, frames)
    ani.save('pulsating_sphere_velocity.gif', dpi=80, writer='imagemagick')

    # Sound pressure
    amplitude = 1e-6  # amplitude of the surface displacement
    impedance_pw = sfs.default.rho0 * sfs.default.c
    max_pressure = omega * impedance_pw * amplitude
    grid = sfs.util.xyz_grid([xmin, xmax], [ymin, ymax], 0, spacing=0.005)
    ani = sound_pressure(
            omega, center, radius, amplitude, grid, frames, pulsate=True,
            colorbar=True, vmin=-max_pressure, vmax=max_pressure)
    ani.save('pulsating_sphere_pressure.gif', dpi=80, writer='imagemagick')
