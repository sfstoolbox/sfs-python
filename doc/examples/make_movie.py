"""Example how to generate an animation from a pre-computed sound field.

p and grid should contain the pressure field and axes of the sound
field, respectively.

"""

import numpy as np
import matplotlib.pyplot as plt
import sfs

# total number of frames
frames = 240

fig = plt.figure(figsize=(15, 15))
for i in range(frames):
    plt.cla()
    ph = sfs.mono.synthesized.shiftphase(p, i / frames * 4 * np.pi)
    sfs.plot.soundfield(2.5e-9 * ph, grid, colorbar=False, cmap=plt.cm.BrBG)
    fname = '_tmp%03d.png' % i
    print('Saving frame', fname)
    plt.savefig(fname)


# mencoder command line to convert PNGs to movie
# mencoder 'mf://_tmp*.png' -mf type=png:fps=30 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o soundfigure.mpg
