"""Example program to demonstrate how to send a multi-channel time series to
LSL."""

import numpy as np
import time
from pylsl import (StreamInfo, StreamOutlet)

import matplotlib
import matplotlib.pyplot as plt
matplotlib.interactive(True)

from mspacman.generator.simulate_pac import PACGenerator
from pytf import FilterBank

info = StreamInfo(name='RAW',
                  type='RawBrainSignal',
                  channel_count=1,
                  nominal_srate=2**14,
                  channel_format='float32',
                  source_id='myuid34234')

# next make an outlet
outlet = StreamOutlet(info)

bw = 30
cf = np.asarray([30, 60, 90, 120, 150])
fb = FilterBank(nch=1, nsamp=2**12, binsize=2**12, decimate_by=1,
                 bandwidth=bw, center_freqs=cf, freq_bands=None, order=2**12, sample_rate=2**14,
                 hilbert=False, domain='time', nprocs=1, mprocs=False)

# fig1, ax1 = plt.subplots(1, 1)
# fig2, ax2 = plt.subplots(cf.size, 1)
# ax2 = np.asarray(ax2).ravel()

# from numpy.fft import fftfreq, fftshift
# w = fftshift(fftfreq(fb.filts.size)) * 2**14
# print(fb.domain, fb.order)
# plt.figure()
# plt.plot(w, np.abs(fb.filts))
# plt.xlim([-100, 100])
print("now sending data...")

i = 0
tstart = 0
tdur = 2**12
while True:
    j = np.abs(np.sin(2*np.pi*i/50))
    pacgen = PACGenerator(16, 130, 1, .25, sample_rate=2**14, phase_amp=np.pi, seed=False)
    mysample = pacgen.simulate(tdur, j, nch=1, noise=False)

    tstart += 500
    n = np.arange(tstart, tstart+tdur)

    # now send it and wait for a bit
    # time.sleep(0.001)
    outlet.push_chunk(mysample)

    tmp = fb.analysis(mysample, window='hanning')
    print(mysample.shape, tmp.shape)

    i+=1

    # try:
    #     # ax1.clear()
    #     # ax1.plot(n, mysample.T)
    #     # ax1.set_ylim([-3, 3])
    #
    #     for (i,), ax in np.ndenumerate(ax2):
    #         ax.clear()
    #         ax.plot(n, tmp[:,i,:].T)
    #
    #         ax.set_ylim([-1, 1])
    #         fig2.tight_layout()
    #
    #     plt.pause(0.00001)
    #
    # except KeyboardInterrupt:
    #     plt.show(block=True)
    #     break
