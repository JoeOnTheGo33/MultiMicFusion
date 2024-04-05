import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

data1, samplerate1 = sf.read('normalized/sensor_1.wav')
data2, samplerate2 = sf.read('normalized/sensor_2.wav')
data3, samplerate3 = sf.read('normalized/sensor_3.wav')

assert(samplerate1 == samplerate2)
assert(data1.shape == data2.shape)

data1 = data1 / np.mean(np.abs(data1))
data2 = data2 / np.mean(np.abs(data2))
data3 = data3 / np.mean(np.abs(data3))
X = np.array([data1, data2, data3])

N = len(data1)
n = 20
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
    # W = W / np.linalg.norm(W, axis=0, keepdims=True) # Rescale weights per sensor
    return W


# W = difference_weights()
W = variance_weights()
plt.plot(W, label=["1", "2", "3"])
plt.legend()

output = np.zeros(N)

# i : start of block
# j : end of block
# k : block
for i, j, k in zip(ii[:-1], ii[1:], np.arange(n)):
    W_block = W[k, :].flatten()
    output[i:j] = np.average(X[:, i:j], axis=0, weights=W_block)

output /= 3

sf.write("output.wav", output.flatten(), sr)

plt.figure()
plt.plot(tt.flatten(), output)
plt.plot(tt.flatten(), data1, alpha=.5)

plt.show()