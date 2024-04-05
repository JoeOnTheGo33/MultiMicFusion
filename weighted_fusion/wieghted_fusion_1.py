import soundfile as sf
import noisereduce as nr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

def rms(x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))

data1, samplerate1 = sf.read('normalized/sensor_1.wav')
data2, samplerate2 = sf.read('normalized/sensor_2.wav')
data3, samplerate3 = sf.read('normalized/sensor_3.wav')

assert(samplerate1 == samplerate2)
assert(data1.shape == data2.shape)

data1 = data1 / np.mean(np.abs(data1))
data2 = data2 / np.mean(np.abs(data2))
data3 = data3 / np.mean(np.abs(data3))
X = np.array([data1, data2, data3])
XE = rms(X, axis=1)

N = len(data1)
n = 30
sr = samplerate1
tt = np.arange(N) / sr
ii = np.linspace(0, N, n, dtype=np.int32)

def difference_weights():
    W = []

    for i, j in zip(ii[:-1], ii[1:]):
        d1 = data1[i:j] - data2[i:j]
        d2 = data2[i:j] - data3[i:j]
        d3 = data1[i:j] - data3[i:j]
        v1 = np.sqrt(np.mean(d1**2))
        v2 = np.sqrt(np.mean(d2**2))
        v3 = np.sqrt(np.mean(d3**2))
        W.append((v1 * v3, v2 * v1, v3 * v2))

    W = np.array(W) # Convert to array
    W = W / np.linalg.norm(W, axis=0, keepdims=True) # Rescale weights per sensor
    return W

def variance_weights():
    W = []

    for i, j in zip(ii[:-1], ii[1:]):
        d1 = data1[i:j] - data2[i:j]
        d2 = data2[i:j] - data3[i:j]
        d3 = data1[i:j] - data3[i:j]
        v1 = np.var(d1)
        v2 = np.var(d2)
        v3 = np.var(d3)
        W.append((v1 * v3, v2 * v1, v3 * v2))

    W = np.array(W) # Convert to array
    # Rescale weights per sensor so the sensors are farily weighted which each other
    # W = W / np.linalg.norm(W, axis=0, keepdims=True)
    return W


# W = difference_weights()
W = variance_weights()
plt.plot(W, label=["1", "2", "3"])
plt.legend()

output = np.zeros(N)

# i : start of window
# j : end of window
# k : window
for i, j, k in zip(ii[:-1], ii[1:], np.arange(n)):
    W_window = W[k, :].flatten()
    X_window = X[:, i:j]
    X_clean = np.zeros(X_window.shape)
    for s in range(3):
        X_clean[s, :] = nr.reduce_noise(y = X_window[s, :], y_noise=X_window[2,:], sr=sr, n_std_thresh_stationary=2.0,stationary=True)
    output[i:j] = np.average(X_clean, axis=0, weights=W_window)

output = output / np.max(np.abs(output)) * .7

sf.write("weighted_fusion/output.wav", output.flatten(), sr)
print("DONE")

plt.figure()
plt.plot(tt.flatten(), output)
plt.plot(tt.flatten(), data1, alpha=.5)

plt.show()