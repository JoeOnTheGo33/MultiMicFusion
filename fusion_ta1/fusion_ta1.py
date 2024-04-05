import torchaudio
import noisereduce as nr
from torchaudio.functional import *
import torch as tr
import numpy as np

# pip install torchaudio

x1, sr = torchaudio.load('normalized/sensor_1.wav')
x2, sr = torchaudio.load('normalized/sensor_2.wav')
x3, sr = torchaudio.load('normalized/sensor_3.wav')
X = tr.concat([x1, x2, x3], dim=0)


N = len(x1[0])
n = 3
tt = np.arange(N) / sr
ii = np.linspace(0, N, n, dtype=np.int32)



filtered_waveform = highpass_biquad(x1 + x2 + x3, sr, 5.0, .707)

filtered_waveform = tr.tensor(nr.reduce_noise(y = filtered_waveform, sr=sr, n_std_thresh_stationary=.7,stationary=False))

print(filtered_waveform)

torchaudio.save('fusion_ta1/output2.wav', filtered_waveform, sr)
