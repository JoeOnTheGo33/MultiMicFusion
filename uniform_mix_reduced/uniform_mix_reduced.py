import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt

data1, samplerate1 = sf.read('normalized/sensor_1.wav')
data2, samplerate2 = sf.read('normalized/sensor_2.wav')
data3, samplerate3 = sf.read('normalized/sensor_3.wav')

# Mix audio. Scale isn't really important; it just changes the volume
data = (data1 + data2 + data3)

reduced_noise = nr.reduce_noise(y = data, sr=samplerate1, n_std_thresh_stationary=2.0,stationary=True)
sf.write("uniform_mix_reduced/uni_mix_reduced_output.wav", reduced_noise, samplerate1)
# Plot
plt.plot(data, label="Input")
plt.plot(reduced_noise, label="Output")
plt.legend()
plt.title("Stationary Noise Reduction (Uniform Mix)")
plt.show()