import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt


data, samplerate = sf.read('bird_song_with_noise.wav')

# Stereo to Mono
data = (data[:,0] + data[:,1]) / 2
reduced_noise = nr.reduce_noise(y = data, sr=samplerate, n_std_thresh_stationary=1.5,stationary=True)
sf.write("processed.wav", reduced_noise, samplerate)

# Plot
plt.plot(data)
plt.plot(reduced_noise)
plt.show()