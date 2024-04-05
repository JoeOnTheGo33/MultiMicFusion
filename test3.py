import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt


data1, samplerate1 = sf.read('synced-wav/sensor_1.wav')
data2, samplerate2 = sf.read('synced-wav/sensor_2.wav')
assert(samplerate1 == samplerate2)
samplerate = samplerate1

# Try subtraction
data = (data1 - data2) / 2

# reduced_noise = nr.reduce_noise(y = data, sr=samplerate, n_std_thresh_stationary=1.5,stationary=True)

sf.write("test3.wav", data, samplerate)

# Plot
plt.plot(data)
plt.show()