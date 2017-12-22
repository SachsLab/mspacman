import numpy as np
import matplotlib.pyplot as plt

import warnings
from pytf.filter.filterbank import FilterBank
from .algorithm.pac_ import (pad, mrpad, polar, pac_mvl, pac_hr, pac_mi)
from .utilities.parallel import (Parallel, ParallelDummy)

func = dict([
    ('mi', mrpad),
    ('hr', mrpad),
    ('mvl', polar)
])
pacfunc = dict([
    ('mi', pac_mi),
    ('hr', pac_hr),
    ('mvl', pac_mvl)
])
class PhaseAmplitudeCoupling(object):
    """ This class is for computing phase-amplitude coupling (PAC).

    Parameters:
    -----------
    freq_phase: ndarray
        An array of frequency bands for phase signals.

    freq_amp: ndarray
        An array of frequency bands for amplitude signals.

    sample_rate: int
        Sampling of the input signal.

    nprocs: int
        Number of processes to enable for multiprocessing.

    unit: str (default: 'hz')
        The units of all frequencies.

    pac: str (default: 'mi')
        The PAC method specified.

    decimate_by:

    kwargs:

    """
    def __init__(self, nch=1, nsamp=2**11, binsize=2**14, freq_phase=None, freq_amp=None, sample_rate=None, \
                    decimate_by=1, nprocs=1, pac='mi', mprocs=False, **kwargs):

        self._nprocs = nprocs
        self._mprocs = True if self.nprocs > 1 else mprocs 
        self._method = pac

        self._decimate_by = decimate_by
        self._nsamp = nsamp
        self._nsamp_ = self.nsamp // self.decimate_by
        self.overlap_factor = 0.5
        self._binsize = binsize
        self._nwin = int((self._nsamp / self._binsize) / self.overlap_factor + 1)

        # Initialize filter bank parameters
        self._freq_phase = freq_phase
        self._freq_amp = freq_amp
        self._sample_rate = sample_rate

        self.nch = nch
        self.fpsize = self.freq_phase.shape[0]
        self.fasize = self.freq_amp.shape[0]

        if self.freq_phase is not None or self.freq_amp is not None:
            self._order = self._binsize // 4
            self._los = FilterBank(nch=self.nch, nwin=self._nwin,\
                    binsize=self._binsize, freq_bands=self.freq_phase,\
                    order=self._order, sample_rate=self.sample_rate,\
                    decimate_by=self.decimate_by, hilbert=True, nprocs=1\
                )

            self._his = FilterBank(nch=self.nch, nwin=self._nwin,\
                    binsize=self._binsize, freq_bands=self.freq_amp,\
                    order=self._order, sample_rate=self.sample_rate,\
                    decimate_by=self.decimate_by, hilbert=True, nprocs=1\
                )

        # Initialize PAC
        self.init_multiprocess(
                                ins_shape=[(self.nch, self.fpsize, self.nsamp_), (self.nch, self.fasize, self.nsamp_)],\
                                out_shape=(self.nch, self.fpsize, self.fasize), \
                                method=self.method, **kwargs
                        )

    def init_multiprocess(self, ins_shape=None, out_shape=None,
                                ins_dtype=[np.float32, np.float32],
                                out_dtype=np.float32, **kwargs):
        """ Initializing multiprocessing for this class.

        Parameters:
        ----------
        ins_shape: tuple (default: None)
            The shape of the input arrays, phases, and amplitudes.

        out_shape: tuple (default: None)
            The shape of the output array (nch, nlo, nhi, nbins).
        """
        if self.mprocs:
            self._pfunc = Parallel(
                            self.get_pac, nprocs=self.nprocs, axis=1, flag=0,
                            ins_shape=ins_shape, out_shape=out_shape,
                            ins_dtype=ins_dtype, out_dtype=out_dtype,
                            **kwargs
                        )

        else:
            warnings.warn("The multiprocessing is disabled! To enable multiprocessing, "+\
                        "specify 'ins_shape' and 'out_shape' for preallocating shared memory.")

            self._pfunc = ParallelDummy(self.get_pac, **kwargs)


    def kill(self, opt=None): # kill the multiprocess
        """ Killing all the multiprocessing processes.
        """
        self._pfunc.kill(opt=True)
        # self._los._pfunc.kill(opt=True)
        # self._his._pfunc.kill(opt=True)

    def comodulogram(self, x=None, x_lo=None, x_hi=None, **kwargs):
        """ Compute vectorized PAC. i.e., the computed PAC has the shape of (nch, nlo, nhi).

        Parameters:
        -----------
        x: ndarray (nch, nsamp)
            Raw input signal.

        x_lo: ndarray (nch, nfreq, nsamp)
            Lowpass filtered input signal.

        x_hi: ndarray (nch, nfreq, nsamp)
            Highpass filtered input signal.

        kwargs: dict
            The key-word arguments to the functions, pad or polar.

        Return:
        -------
        The computed PAC.
        """
        self._xlo = x_lo if x_lo is not None else self.los.analysis(x, window='hanning')
        self._xhi = x_hi if x_hi is not None else self.his.analysis(x, window='hanning')

        self._comod = self._pfunc.result(np.angle(self._xlo), np.abs(self._xhi))
        return self._comod

    def plot_comodulogram(self, ch=None, axs=None, figsize=None, cbar=False, cmap=None,
                                vmin=None, vmax=None, norm=None, label=False):
        """
        Plot comodulogram.

        Parameters:
        -----------

        Returns:
        --------
        """
        _comod = self._comod[ch,:,:][np.newaxis,:,:] if ch is not None else self._comod

        nch, nlo, nhi = _comod.shape

        # Build Figures
        figsize = (4 * nch, 5) if figsize is None else figsize
        if axs is None:
            fig, axs = plt.subplots(1, nch, figsize=figsize)
        else:
            fig = axs[0].figure

        self._axs = np.array(axs).ravel()
        self._fig = fig

        # Imshow Parameters
        extent = [  self.freq_phase[0,0], self.freq_phase[-1,-1],
                    self.freq_amp[0,0], self.freq_amp[-1,-1]]

        cmap = None if str(cmap).lower() is None else cmap
        self._vmin = np.min(self._comod) if vmin is None else vmin
        self._vmax = np.max(self._comod) if vmax is None else vmax

        if norm is 'log':
            from matplotlib.colors import LogNorm
            norm = LogNorm(vmin=vmin, vmax=vmax)
            self._vmin = None
            self._vmax = None

        elif norm.lower() is 'none':
            norm = None

        # Plot imshow()
        self.cax = self._axs.copy()
        for (i,), ax in np.ndenumerate(self._axs):
            self.cax[i] = ax.imshow(_comod[i,:,:].T, cmap=cmap, norm=norm, vmin=self._vmin, vmax=self._vmax,
                            aspect='auto', origin='lower', extent=extent,
                            interpolation=None)

        # Plot Labels
        if label:
            self._axs[0].set_ylabel('Amp. Freqs. [{}]'.format('hz'.title()))
            for ax in self._axs:
                ax.set_xlabel('Phase Freqs. [{}]'.format('hz'.title()))

        return self._fig

    def plot_pad(self, ch=None, axs=None, figsize=None, colors=None, nbins=10):
        """
        Plot PAD.

        Parameters:
        -----------

        Returns:
        --------
        """
        _lo = self._xlo[ch,:,:][np.newaxis,:,:] if ch is not None else self._xlo
        _hi = self._xhi[ch,:,:][np.newaxis,:,:] if ch is not None else self._xhi

        xlo = _lo.mean(axis=1)[:,np.newaxis,:]
        xhi = _hi.mean(axis=1)[:,np.newaxis,:]

        pd = self._pac_repr_func(np.angle(xlo), np.abs(xhi), nbins=nbins)[:,0,0,:]

        nch, _ = pd.shape
        bin_centers = np.linspace(-np.pi, np.pi-np.pi/50, nbins+1) + np.pi/10

        if axs is None:
            fig, axs = plt.subplots(1, nch, figsize=figsize)
        else:
            fig = axs[0].figure

        self._axs = np.array(axs).ravel()
        self._fig = fig
        for (i,), ax in np.ndenumerate(self._axs):
            ax.bar(bin_centers[:-1], pd[i,:], width=.5)

        return self._fig

    @staticmethod
    def get_pac(ang, amp, method='mi', **kwargs):
        """
        """
        return pacfunc[method](func[method](ang, amp, **kwargs))

    # ===================================
    # Define Setters Parameters
    # ===================================
    @property
    def los(self):
        return self._los

    @property
    def his(self):
        return self._his

    @property
    def freq_phase(self):
        return self._freq_phase

    @property
    def freq_amp(self):
        return self._freq_amp

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def method(self):
        return self._method

    @property
    def nprocs(self):
        return self._nprocs

    @property
    def unit(self):
        return self._unit

    @property
    def mprocs(self):
        return self._mprocs

    @property
    def nsamp(self):
        return self._nsamp

    @property
    def nsamp_(self):
        return self._nsamp_

    @property
    def decimate_by(self):
        return self._decimate_by
