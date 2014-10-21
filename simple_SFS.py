# example thats illustrates the application of python to the numeric simulation of SFS systems
#
# S.Spors, 21.10.14

import matplotlib.pyplot as plt
import math
import numpy as np
import SFS


# parameters
dx = 0.1       			# secondary source distance
N = 50         			# number of secondary sources
pw_angle = math.pi/4    # incidence angle of plane wave
f = 500        			# frequency


# wavenumber
k = 2*math.pi*f/343;

# spatial grid
x = np.arange(-5, 5, 0.02)        
y = np.arange(0, 5, 0.02)



# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------

# generate tapering window
twin = SFS.tapering.weight(N)

# compute synthesized sound field
z = 0
for n in range(0,N-1):
    pos = (n - N/2 + 1/2) * dx, 0
    #pos = ((-N//2+n)*dx,0)
    z = z + SFS.drivingfunction.pw_delay(k, pos, pw_angle) * twin[n] * SFS.source.point(k, pos, x, y)

# plot synthesized sound field
plt.figure(figsize = (15, 15))

plt.imshow(np.real(z), cmap=plt.cm.RdBu, origin='lower', extent=[-5,5,0,5], vmax=10, vmin=-10, aspect='equal')
plt.colorbar()

plt.savefig('soundfield.png')
