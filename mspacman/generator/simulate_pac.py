import numpy as np
from .noise import (white, pink)
from pyfftw.interfaces.numpy_fft import irfft, fft, fftfreq, fftshift
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)

class PACGenerator:
    """
    Define the parameters for generating a Phase-Amplitude coupling (PAC) signal.
    Parameters
    ----------
    fp: int
        Frequency for the phase-giving signal.
    fa: int
        Frequency for the Amplitude-enveloped signal.
    sp: float
        Amplitude for the phase-giving signal.
    sa: float
        Amplitude for the Amplitude-enveloped signal.
    """

    def __init__(self, freq_phase, freq_amp, scale_phase, scale_amp, phase_amp=0, sample_rate=None, seed=True):

        self._freq_phase = freq_phase
        self._freq_amp = freq_amp

        self._scale_phase = scale_phase
        self._scale_amp = scale_amp

        self._phase_amp = phase_amp

        self.sample_rate = sample_rate

        if seed:
            np.random.seed(0)

    def simulate(self, n, pac, nch=None, noise=None):
        """ Generate a multi-channel PAC signal.
        Parameters
        ----------
        pac: float or list
            The intensity of Phase-Amplitude coupling.
        num_channel: int, optional
            The number of channels for the generated signal.
        Returns
        -------
        signal: array_like
            Simulated PAC signal(s).
        """
        if not isinstance(pac, np.ndarray):
            self.pac = np.asarray(pac)

        self.nch = nch if nch is not None else self.pac.size
        if self.nch != self.pac.size:
            self.pac = self.pac.repeat(self.nch)

        if not hasattr(n, '__iter__'):
            n = np.arange(int(n))

        N = n[-1]

        # Pre-allocate memory for the arrays
        lo, hi = self._pac_hr(n, self.pac, self.scale_phase, self.scale_amp, \
                                        self.freq_phase, self.freq_amp, self.sample_rate, phase_amp=self.phase_amp)

        def noise_func(*args, **kwargs):
            return white(*args, **kwargs)

        noise_ = 0 if noise is None or noise is False else noise_func(self.nch, N+1, std=.5)

        return lo + hi + noise_

    @staticmethod
    def _pac_hr(n, pac, scale_phase, scale_amp, freq_phase, freq_amp, sample_rate, phase_amp=0):
        """ Generate the PAC signal controlled ideal for height ratios.
        Parameters
        ----------
        pac: float
            The intensity of Phase-Amplitude coupling.
        Returns
        -------
        sig: array_like
            An array of coupled signal generated.
        """
        if not hasattr(n, '__iter__'):
            n = np.arange(n)
        n = np.atleast_2d(n)

        pac = pac if hasattr(pac, '__iter__') else [pac]
        pac = np.atleast_2d(pac).T

        freq_phase = freq_phase if hasattr(freq_phase, '__iter__') else [freq_phase]
        freq_amp = freq_amp if hasattr(freq_amp, '__iter__') else [freq_amp]

        freq_phase = np.atleast_2d(freq_phase).T
        freq_amp = np.atleast_2d(freq_amp).T

        lo = scale_phase * np.sin( 2 * np.pi * freq_phase * n / sample_rate)
        moda = np.sin(2 * np.pi * freq_phase * n / sample_rate + phase_amp)
        ampa = scale_amp * (pac * moda + 2 - pac)
        hi = ampa * np.sin(2 * np.pi * freq_amp * n / sample_rate)
        return lo, hi

    @property
    def freq_phase(self):
        return self._freq_phase

    @property
    def freq_amp(self):
        return self._freq_amp

    @property
    def scale_phase(self):
        return self._scale_phase

    @property
    def scale_amp(self):
        return self._scale_amp

    @property
    def phase_amp(self):
        return self._phase_amp
