import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def _plot_comodulogram(comod, axs=None, figsize=None, cbar=False, title=None,
                       label=False, xlabel=False, ylabel=False, xaxis=None, yaxis=None,
                       fontsize={'ticks': 15, 'axis': 15, 'title': 20}, **kwargs):

    """
    Plot comodulogram.

    Parameters:
    -----------
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

    _comod = comod
    nlo, nhi = _comod.shape
    xaxis = np.arange(0, nlo) if xaxis is None else np.asarray(xaxis, dtype=np.int32)
    yaxis = np.arange(0, nhi) if yaxis is None else np.asarray(yaxis, dtype=np.int32)

    # Build Figures
    figsize = (4, 5) if figsize is None else figsize
    if axs is None:
        _fig, _ax = plt.subplots(1, 1, figsize=figsize)
    else:
        _ax = axs
        _fig = _ax.figure

    # Plot imshow()
    im = _ax.imshow(_comod.T, aspect='auto', origin='lower',**kwargs)

    # Plot Labels
    xlabel = True if label else xlabel
    ylabel = True if label else ylabel

    if xlabel:
        # x-axis
        tmp = np.where(xaxis%5==0)[0][::2]
        _ax.set_xticks(tmp)
        _ax.set_xticklabels(xaxis[tmp])
        _ax.set_xlabel('Phase Freqs. [{}]'.format('hz'.title()), fontsize=fontsize['axis'])
        for tick in _ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize['ticks'])
    else:
        _ax.set_xticks([])

    if ylabel:
        # y-axis
        tmp = np.where(yaxis%5==0)[0][::2]
        _ax.set_yticks(tmp)
        _ax.set_yticklabels(yaxis[tmp])
        _ax.set_ylabel('Amp. Freqs. [{}]'.format('hz'.title()), fontsize=fontsize['axis'])
        for tick in _ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize['ticks'])
    else:
        _ax.set_yticks([])

    if cbar:
        divider = [make_axes_locatable(_ax)]
        cax = [div.append_axes("right", size="5%", pad=0.1) for div in divider]
        cbar = _fig.colorbar(im, cax = cax[-1])
        cbar.ax.tick_params(labelsize=fontsize['ticks'])

    if title is not None:
        _ax.set_title(title, fontsize=fontsize['title'])

    return _fig
