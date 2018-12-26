"""
Particle displacement of a pulsating sphere.
"""
import sfs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation, patches

# Pulsating sphere
x0 = [0, 0, 0]  # position
a = 0.25  # radius
d_peak = 0.05  # peak value of the surface displacement
f = 1000  # frequency
omega = 2 * np.pi * f  # angular frequency
ka = sfs.util.wavenumber(omega) * a

# Temporal sampling for animation
fs = f * 10  # sampling frequency
L = int(np.round(fs / f))  # number of frames corresponding to one period
t = np.arange(L) / fs  # time

# Uniform grid
N = 120
xmin, xmax = -1, 1
ymin, ymax = -1, 1
x = np.tile(np.linspace(xmin, xmax, N), N)
y = np.repeat(np.linspace(ymin, ymax, N), N)
z = 0
grid = sfs.util.XyzComponents([x, y, z])

# Particle displacement
d = sfs.mono.source.pulsating_sphere_displacement(omega, x0, a, d_peak, grid)

# Animation
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(xlim=(-xmax, xmax), ylim=(-ymax, ymax))
circ_patch = ax.add_patch(patches.Circle(x0[:2],
                                         radius=a,
                                         facecolor='RoyalBlue',
                                         linestyle='None'))
line, = ax.plot(grid[0], grid[1],
                marker='o',
                markersize=1.5,
                markerfacecolor='gray',
                markeredgecolor='None',
                linestyle='None',
                alpha=0.75)
text = ax.text(0.9 * xmax, 0.9 * ymin, '<t>',
               fontsize=16,
               horizontalalignment='right',
               verticalalignment='center')
ax.set_title('$a={:0.2f}$ m, $f={:g}$ Hz ($ka={:0.1f}$)'
             .format(a, f, ka), fontsize=16)
ax.set_xlabel('$x$ / m')
ax.set_ylabel('$y$ / m')
ax.axis('off')


def animate(i, d_peak, line, circ_patch, text):
    phase_shift = np.exp(1j * omega * t[i])
    X = grid + np.real(d * phase_shift)
    line.set_data(X[0], X[1])
    circ_patch.set_radius(a + d_peak * np.real(phase_shift))
    text.set_text('{:0.2f} ms'.format(t[i] * 1000))
    return line,


ani = animation.FuncAnimation(fig,
                              animate,
                              frames=np.arange(L),
                              fargs=(d_peak, line, circ_patch, text))
ani.save('pulsating_sphere.gif', fps=10, dpi=80, writer='imagemagick')
