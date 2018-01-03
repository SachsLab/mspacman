import numpy as np
import matplotlib.pyplot as plt

from pytf import FilterBank
from ..algorithm.pac_ import (pad, mrpad, polar, pac_mvl, pac_hr, pac_mi)
from ..algorithm.blob_ import (detect_blob)
from ..utilities.parallel import (Parallel, ParallelDummy)
from ..viz.pac_plots import (_plot_comodulogram)
# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : BSD (3-clause)

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
    nch: int
        The number of channels of the processing signal.

    nsamp: int
        The sample size of the processing signal.

    freq_phase: ndarray
        An array of frequency bands for phase signals.

    freq_amp: ndarray
        An array of frequency bands for amplitude signals.

    sample_rate: int
        Sampling of the input signal.

    nprocs: int
        Number of processes to enable for multiprocessing.

    pac: str
        The PAC method specified. Default is 'mi'.

    compression_factor:

    kwargs:

    """
    def __init__(self, nch=1, nsamp=2**11, binsize=2**14, freq_phase=None, freq_amp=None, sample_rate=None, \
                    nprocs=1, pac='mi', mprocs=False, **kwargs):

        # Pre-defined parameters
        _comp_threshold = 15
        _overlap_factor = 0.5

        # Phase-Amplitude Coupling Properties
        self._method = pac

        self._freq_phase = freq_phase if isinstance(freq_phase, np.ndarray) else np.asarray(freq_phase)
        self._freq_amp = freq_amp if isinstance(freq_amp, np.ndarray) else np.asarray(freq_amp)
        self._sample_rate = sample_rate
        self.fpsize = self.freq_phase.shape[0]
        self.fasize = self.freq_amp.shape[0]

        # Signal Properties
        _comp_ratio = int(np.floor(self.sample_rate / self.freq_amp[-1,-1] / 2))
        self._compression_factor = _comp_ratio if _comp_ratio < _comp_threshold else _comp_threshold

        self._nch = nch
        self._nsamp = nsamp
        self._nsamp_ = self.nsamp // self.compression_factor

        # Overlap-Window Parameters
        self._binsize = binsize
        self._nwin = int((self._nsamp / self._binsize) / _overlap_factor + 1)

        # Initialize filter bank parameters
        if self.freq_phase is not None or self.freq_amp is not None:
            self._order = self._binsize // 4
            self._los = FilterBank(
                    nch = self.nch,
                    nsamp = self.nsamp,
                    binsize = self._binsize,
                    order = self._order,
                    freq_bands = self.freq_phase,
                    sample_rate = self.sample_rate,
                    decimate_by = self.compression_factor, hilbert=True, nprocs=1
                )

            self._his = FilterBank(
                    nch = self.nch,
                    nsamp = self.nsamp,
                    binsize = self._binsize,
                    order = self._order,
                    freq_bands = self.freq_amp,
                    sample_rate = self.sample_rate,
                    decimate_by = self.compression_factor, hilbert=True, nprocs=1
                )

        # Initialize PAC
        self._nprocs = nprocs
        self._mprocs = True if self.nprocs > 1 else mprocs
        self._pfunc = Parallel(
                        self.get_pac, nprocs=self.nprocs, axis=1, flag=0,
                        ins_shape = [(self.nch, self.fpsize, self.nsamp_),
                                     (self.nch, self.fasize, self.nsamp_)],
                        out_shape = (self.nch, self.fpsize, self.fasize),
                        ins_dtype = [np.float32, np.float32],
                        out_dtype = np.float32,
                        method = self.method,
                        **kwargs
                    ) if self.mprocs else ParallelDummy(self.get_pac, method=self.method, **kwargs)


    def kill(self, opt=None): # kill the multiprocess
        """ Killing all the multiprocessing processes.
        """
        self._pfunc.kill(opt=True)

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

        kwargs: various
            The key-word arguments for 'mrpad' or 'polar'. See '.algorithm.pac_.py'

        Return:
        -------
        The computed PAC.
        """
        self._xlo = x_lo if x_lo is not None else self.los.analysis(x, window='hanning')
        self._xhi = x_hi if x_hi is not None else self.his.analysis(x, window='hanning')

        self._comod = self._pfunc.result(np.angle(self._xlo), np.abs(self._xhi))
        return self._comod

    def plot_comodulogram(self, ch=None, axs=None, figsize=None, cbar=False,
                                title=None, label=False, xlabel=False, ylabel=False,
                                fontsize={'ticks': 15, 'axis': 15, 'title': 20}, **kwargs):
        """
        Plot comodulogram.

        Parameters:
        -----------
        ch: int
            Select the channel for plotting the comodulogram. Default: Select all channels.

        axs: ndarray
            Provide the Matplotlib Axes class to plot on. Default: Create a new figure and axes.

        figsize: tuple
            Specify the figure size.

        label: bool
            Label the plots. Default is False.

        Returns:
        --------
        The matplotlib figure object.
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

        if not isinstance(title, list):
            title = list([title]) * len(self._axs)

        xlabel = True if label else xlabel
        ylabel = True if label else ylabel

        for (i,), ax in np.ndenumerate(self._axs):
            cbar_ = True if i==len(self._axs)-1 and cbar else False
            ylabel_ = True if i==0 and ylabel else False
            _plot_comodulogram(_comod[i,:,:], axs=ax, cbar=cbar_, title=title[i],\
                                xlabel=xlabel, ylabel=ylabel_, xaxis=self._los.center_freqs.squeeze(), yaxis=self._his.center_freqs.squeeze(),\
                                vmin=np.min(_comod), vmax=np.max(_comod), fontsize=fontsize, **kwargs)
        return self._fig

    def plot_pad(self, ch=None, axs=None, figsize=None, colors=None, nbins=10):
        """
        Plot PAD.
        TODO: Need more work on efficiency, as well as various plotting types.

        Parameters:
        -----------
        ch: int
            Select the channel for plotting the comodulogram. Default: Select all channels.

        axs: ndarray
            Provide the Matplotlib Axes class to plot on. Default: Create a new figure and axes.

        figsize: tuple
            Specify the figure size.

        colors:
            Specify the color of the bar.

        kwargs

        Returns:
        --------
        The matplotlib figure object.
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
    def find_pac_blob(comod, threshold=.5, vmin=0, vmax=1):
        return detect_blob(comod, threshold=threshold, vmin=vmin, vmax=vmax)

    @staticmethod
    def get_pac(ang, amp, method='mi', **kwargs):
        """
        Compute PAC from phases and amplitudes of the signal.

        Parameters:
        -----------
        ang: ndarray
            The instantaneous phases of the given signal.

        amp: ndarray
            The instantaneous amplitudes of the given signal.

        method: str
            The PAC method to used. Default: 'mi'.

        kwargs: various
            The key-word arguments for 'mrpad' or 'polar'. See '.algorithm.pac_.py'

        Returns:
        --------
        The computed PAC/comodulogram
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
    def nch(self):
        return self._nch

    @property
    def nsamp(self):
        return self._nsamp

    @property
    def nsamp_(self):
        return self._nsamp_

    @property
    def compression_factor(self):
        return self._compression_factor
