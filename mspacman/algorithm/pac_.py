import numpy as np
from scipy.stats import entropy

def pad(ang, amp, nbins=30, axis=-1):
    """ Organize the signals into a phase-amplitude distribution.

    Parameters
    ----------
    ang: array_like
        Phase of the low frequency signal.

    amp: array_like
        Amplitude envelop of the high frequency signal.

    nbins: int
        The bin size for the phases.

    Returns
    -------
    pd: array_like
        The phase-amplitude distribution.

    phase_bins: array_like
        The binned locations (phases) of the distribution.
    """
    nfr_a = amp.shape[1]
    nch   = ang.shape[0]
    nfr_p = ang.shape[1]
    phase_bins = np.linspace(-np.pi, np.pi, int(nbins + 1))
    pd = np.zeros((nch, nfr_p, nfr_a, int(nbins)))
    for b in range(int(nbins)):
        t_phase = np.logical_and(ang>=phase_bins[b], ang<phase_bins[b+1])
        pd[:,:,:,b] = np.mean(amp[:,np.newaxis,:,:] * t_phase[:,:,np.newaxis,:], axis=axis)

    return pd

def mrpad(ang, amp, nbins=30, axis=-1, flag=0):
    size1 = ang.shape[-1]
    size2 = amp.shape[-1]
    decimate = int(np.round(size2 / size1))

    angd = ang
    ampd = amp
    if flag:
        ampd = ampd[:,:,::decimate]
    else:
        angd = np.repeat(angd, decimate, axis=axis)
        diff = angd.shape[-1] - ampd.shape[-1]
        if diff:
            angd = angd[:,:,:-diff]

    return pad(angd, ampd, nbins=nbins, axis=axis)

def polar(ang, amp, normalize=True):
    """ Calculate the polar coordinates of the amplitude and the phase as time changes.

    Parameters
    ----------
    ang: array_like
        Phase of the low frequency signal.

    amp: array_like
        Amplitude envelop of the high frequency signal.

    Returns
    -------
    z: array_like
        The complex exponentials of the signal.

    Note
    ----
    The input signals can only be 1-dimensional (along the number of samples).

    """
    ang = ang[:,:,np.newaxis,:]
    amp = amp[:,np.newaxis,:,:]
    z = amp * np.exp(1j * ang)

    if normalize:
        z /= np.max(np.abs(z))
    return z

def pac_mvl(z):
    """ Calculate PAC using the mean vector length.

    Parameters
    ----------
    ang: array_like
        Phase of the low frequency signal.

    amp: array_like
        Amplitude envelop of the high frequency signal.

    Returns
    -------
    out: float
        The pac strength using the mean vector length.

    Note
    ----
    The input signals can only be 1-dimensional (along the number of samples).

    """
    # out = np.abs(np.mean(z, axis=-1))
    # out = np.abs(np.sum(z,axis=0))
    # out /= np.sqrt(np.sum(amp * amp,axis=0))
    # print(z.shape, out, np.max(np.abs(z)), np.mean(amp, axis=0))

    # out /= np.max(amp)
    # out /= np.sqrt(z.shape[0])
    return np.abs(np.mean(z, axis=-1))

def pac_hr(pd):
    """ Calculate PAC value using the height ratio.

    Parameters
    ----------
    ang: array_like
        Phase of the low frequency signal.

    amp: array_like
        Amplitude envelop of the high frequency signal.

    Returns
    -------
    The pac strength using the height ratio.

    Note
    ----
    The input signals can only be 1-dimensional (along the number of samples).

    """
    return 1 - np.nanmin(pd, axis=-1) / np.nanmax(pd, axis=-1)

def pac_mi(pd):
    """ Calculate PAC using the modulation index.

    Modulation Index
    See Adriano et al., J Neurophysiol 2010 for details.
    Dkl(P, U) = sum(P * log(P/U)),
    where P is phase-amplitude-distribution,
          U is uniform distribution,
          Dkl is Kullback-Liebler distance

    MI = Dkl(P, U)/log(N)
      Where N is the number of phase bins

    Parameters
    ----------
    ang: array_like
        Phase of the low frequency signal.

    amp: array_like
        Amplitude envelop of the high frequency signal.

    Returns
    -------
    The pac strength using the modulation index.

    Note
    ----
    The input signals can only be 1-dimensional (along the number of samples).

    """
    return entropy(pd.T, np.ones(pd.shape).T).T
