import torchaudio
import noisereduce as nr
from torchaudio.functional import *
import torch as tr
import numpy as np
import matplotlib.pyplot as plt

# %%
def moving_average(samples, window_size):
    # samples: (..., time)
    y = []
    for j in range(samples.shape[0]):
        for i in range(-window_size//2, window_size//2):
            y.append(tr.roll(samples[j], i, dims=0))
    return tr.mean(tr.row_stack(y), dim=0, keepdim=True)

# %%
x1, sr = torchaudio.load('normalized/sensor_1.wav')
x2, sr = torchaudio.load('normalized/sensor_2.wav')
x3, sr = torchaudio.load('normalized/sensor_3.wav')
X = tr.concat([x1, x2, x3], dim=0)

# N = len(x1[0])
# n = 3
# tt = np.arange(N) / sr
# ii = np.linspace(0, N, n, dtype=np.int32)

filtered_waveform = moving_average(X, 5)

print(filtered_waveform)

torchaudio.save('output.wav', filtered_waveform, sr)

# %%
plt.plot(filtered_waveform[0, :100000])