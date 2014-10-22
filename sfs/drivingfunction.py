import numpy as np


def pw_delay(k,position,angle):

    d=np.exp(-1j*k*(np.cos(angle)*position[0] + np.sin(angle)*position[1]))

    return d

