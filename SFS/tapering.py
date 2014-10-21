from scipy import signal


def weight(N): 

	twin = signal.kaiser(N,2);

	return twin
