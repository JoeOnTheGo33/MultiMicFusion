import noisereduce as nr
import librosa
import matplotlib.pyplot as plt

data, samplerate = librosa.load('bird_song_with_noise.wav')

print(data.shape, samplerate)

# Stereo to Mono
data = (data[:,0] + data[:,1]) / 2
reduced_noise = nr.reduce_noise(y = data, sr=samplerate, n_std_thresh_stationary=1.5,stationary=True)
librosa.write("processed.wav", reduced_noise, samplerate)

# Plot
plt.plot(data)
plt.plot(reduced_noise)
plt.show()