# MSPACMan

This Python package provides tools to estimate **phase-amplitude coupling (PAC)** in neural time series.

The development of this project is strongly influenced by a couple of existing PAC analysis tools, [pacpy](https://github.com/voytekresearch/pacpy) and [pactools](https://github.com/pactools/pactools).
However, MSPACMan is developed due to the need for implementing PAC analysis to real-time applications, it is thus by no means a replacement of the existing tools.

## Binder
A [Binder](https://mybinder.org) of the analysis using this work is provided [here](https://github.com/davidlu89/notes_mspacman).

## Dependencies
* [numpy](http://www.numpy.org): The Python library used for efficient manipulation multi-dimensional array.
* [scipy](https://www.scipy.org): The Python library used for scientific computing utilizing the unique NumPy structure in an optimized way. 
* [matplotlib](https://matplotlib.org): The Python library used for visualizing and plotting of data.
* [pyfftw](https://github.com/pyFFTW/pyFFTW): A Python wrapper around [FFTW](http://www.fftw.org), the speedy FFT library.
* [pytf](https://github.com/davidlu89/pytf): A Python library designed for performing time frequency analysis.

## Cite this work
If you use this code in your project, please cite [Lu et al. 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0204260):

@article {lu2018,
    author = {David Chao-Chia Lu, Chadwick Boulay, Adrian D.C. Chan, Adam J. Sachs},
    title = {Realtime phase-amplitude coupling analysis of micro electrode recorded brain signals},
    year = {2018},
    doi = {},
    publisher = {PlosOne},
    URL = {},
    journal = {PlosOne}
}
