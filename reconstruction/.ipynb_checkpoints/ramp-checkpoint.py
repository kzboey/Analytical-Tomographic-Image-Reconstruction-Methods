import scipy
from scipy.fft import fft, ifft
import numpy as np

"""
    Implement the filter in (3.4.14)
"""
def ramp_filter(size):
    n = np.concatenate(
        (
            # increasing range from 1 to size/2, and again down to 1, step size 2
            np.arange(1, size / 2 + 1, 2, dtype=int),
            np.arange(size / 2 - 1, 0, -2, dtype=int),
        )
    )
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    return 2 * np.real(fft(f))[:, np.newaxis]

