import numpy as np
from pyfftw.interfaces.numpy_fft import rfft, irfft, fftfreq
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)

def rms(x, axis=-1):
    return np.sqrt(np.mean(np.abs(x)**2., axis=axis))

def normalize(y, x=None):
    """normalize power in y to a (standard normal) white noise signal.
    Optionally normalize to power in signal `x`.
    #The mean power of a Gaussian with :math:`\\mu=0` and :math:`\\sigma=1` is 1.
    """
    x = 1. if x is None else x
    return (y.T * rms(x) / rms(y)).T

def white(*N, mean=0, std=1):
    """ White noise.

    :param N: Amount of samples.

    White noise has a constant power density. It's narrowband spectrum is therefore flat.
    The power in white noise will increase by a factor of two for each octave band,
    and therefore increases with 3 dB per octave.
    """
    return std * np.random.randn(*N) + mean

def pink(*N, mean=0, std=1):
    """
    =================================
    Pink Noise
    -------------------
    Pink noise has equal power in bands that are proportionally wide.
    Power density decreases with 3 dB per octave.
    =================================
    """
    if len(N) < 2:
        N = tuple((1, N[0]))

    n1, n2 = N
    uneven = n2 % 2

    X = np.random.randn(n1,n2//2+1+uneven) + 1j * np.random.randn(n1,n2//2+1+uneven)

    S = np.sqrt(np.arange(X.shape[-1])+1) # +1 to avoid divide by zero

    y = (irfft(X/S)).real
    if uneven:
        y = y[:,:-1]

    # Normalize the results to the white noise
    y = normalize(y, white(*N, mean=mean, std=std))
    if y.ndim<2:
        return y.flatten()
    else:
        return y
# def white_noise(N, amp=.5):
#     """
#     =================================
#     White Noise
#     -----------------
#     White noise has a constant power density. It's narrowband spectrum is therefore flat.
#     The power in white noise will increase by a factor of two for each octave band,
#     and therefore increases with 3 dB per octave.
#     =================================
#     """
#     return amp * np.random.randn(N)
#
# def pink_noise(N, amp=1):
#     """
#     =================================
#     Pink Noise
#     -------------------
#     Pink noise has equal power in bands that are proportionally wide.
#     Power density decreases with 3 dB per octave.
#     =================================
#     """
#     uneven = N % 2
#     X = np.random.randn(N//2+1+uneven) + 1j * np.random.randn(N//2+1+uneven)
#     S = np.sqrt(np.arange(len(X))+1.) # +1 to avoid divide by zero
#
#     y = (irfft(X/S)).real
#     if uneven: y = y[:-1]
#
#     # Normalize the results to the white noise
#     return amp * y * np.sqrt((np.abs(white_noise(N))**2.0).mean() / (np.abs(y)**2.0).mean()).real
