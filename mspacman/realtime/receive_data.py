"""Example program to show how to read a multi-channel time series from LSL."""
import numpy as np
import time
from buffer_ import Buffer
from pylsl import (StreamInlet, resolve_stream, resolve_byprop)

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
# streams = resolve_stream('type', 'EEG')
streams = resolve_byprop("type", "RawBrainSignal")

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
info = inlet.info()
print(info.nominal_srate(), info.channel_count())
while True:
    chunk, timestamp = inlet.pull_chunk()
    if len(timestamp):
        print(np.asarray(chunk).T.shape)
