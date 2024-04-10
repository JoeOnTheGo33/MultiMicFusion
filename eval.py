import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

def get_signal(filename):
    wav_file = wave.open(filename, 'r')
    signal = wav_file.readframes(-1)
    signal = np.frombuffer(signal, dtype='int16')
    wav_file.close()
    return signal

def get_framerate(filename):
    wav_file = wave.open(filename, 'r')
    framerate = wav_file.getframerate()
    wav_file.close()
    return framerate

def compute_freq(signal, framerate):
    # Compute the FFT of the signal and take the absolute value to get the magnitude
    spectrum = np.abs(fft(signal))

    # Compute the frequencies corresponding to the spectrum and only consider the positive frequencies
    freq = np.fft.fftfreq(len(spectrum), 1 / framerate)
    mask = freq > 0
    spectrum = spectrum[mask]
    freq = freq[mask]
    return freq, spectrum

# create histogram of amplitdues of audio file
def histogram_of_amplitudes(filename, title):
    signal = get_signal(filename)

    # Create a new figure
    plt.figure()

    # Plot the histogram
    plt.hist(signal, bins=100)

    # print highest and lowest amplitude of signal
    print(f'Highest amplitude: {max(signal)}, lowest amplitude: {min(signal)}')

    # Label the axes
    plt.xlabel('Amplitude')
    plt.ylabel('Count')
    plt.title(f'Histogram of amplitudes of {title}')

    plt.show()
    plt.close()
    return 

def histogram_of_frequencies(filename, title):
    signal = get_signal(filename)
    # Get the frame rate
    framerate = get_framerate(filename)

    freq, spectrum = compute_freq(signal, framerate)
    # Create a new figure
    plt.figure()

    # Plot the histogram of the frequencies
    plt.hist(freq, bins=100, weights=spectrum)

    # Label the axes
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')

    # Here you can specify the title
    plt.title(f'Histogram of frequencies of {title}')

    # Save the plot
    plt.show()
    
    return

def histogram_of_volume_db(filename, title):
    # Open the file in read mode
    wav_file = wave.open(filename, 'r')
    # Read frames from the file
    signal = wav_file.readframes(-1)
    signal = np.frombuffer(signal, dtype='int16')
    # Close the file
    wav_file.close()
    # Convert the signal to volume in dB
    volume_db = 20 * np.log10(np.abs(signal) + 10)
    # Remove -inf values
    volume_db = volume_db[volume_db != -np.inf]

    # Create a new figure
    plt.figure()
    # Plot the histogram of the volume in dB
    plt.hist(volume_db, bins=10)
    # Label the axes
    plt.xlabel('Volume [dB]')
    plt.ylabel('Count')
    # Here you can specify the title
    plt.title(f'Histogram of volume (dB) of {title}')
    # Save the plot
    plt.show()
    return

def generate_all_histograms(filename):
    histogram_of_amplitudes(filename)
    histogram_of_frequencies(filename)
    histogram_of_volume_db(filename)
    return

def number_of_samples(filename):
    wav_file = wave.open(filename, 'r')
    framerate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    wav_file.close()
    return num_frames

def count_all_samples():
    for file in ['test_rec.wav', 'processed2.wav', 'sensor_1.wav', 'sensor_2.wav', 'sensor_3.wav', 'uni_mix_reduced_output.wav']:
        print(f'Number of samples in {file}: {number_of_samples(file)}')
        # all samples have 9595771 frames
    return

def manhattan_distance_of_volume(file_one, file_two):
    signal_one = get_signal(file_one)
    signal_two = get_signal(file_two)
    if len(signal_two) > len(signal_one):
        signal_two = signal_two[::2]
    elif len(signal_one) > len(signal_two):
        signal_one = signal_one[::2]

    volume_db_one = 20 * np.log10(np.abs(signal_one)+1) # warning when log 0
    volume_db_two = 20 * np.log10(np.abs(signal_two)+1)
    #volume_db_one = volume_db_one[volume_db_one != -np.inf]
    #volume_db_two = volume_db_two[volume_db_two != -np.inf]
    manhattan_distance = np.sum(np.abs(volume_db_one - volume_db_two))
    print(f'Manhattan distance of volume between {file_one} and {file_two}: {manhattan_distance}')
    return manhattan_distance

def manhattan_distance_amplitude(file_one, file_two):
    signal_one = get_signal(file_one)
    signal_two = get_signal(file_two)

    # remove every second value from signal_two if its longer than signal_one
    if len(signal_one) > len(signal_two):
        signal_one = signal_one[::2]

    manhattan_distance = np.sum(np.abs(signal_one - signal_two))
    manhattan_distance = "{:.3e}".format(manhattan_distance)
    print(f'Manhattan distance of amplitude between {file_one} and {file_two}: {manhattan_distance}')
    return manhattan_distance

def manhattan_distance_frequence(file_one, file_two):
    signal_one = get_signal(file_one)
    signal_two = get_signal(file_two)

    # remove every second value from signal_two if its longer than signal_one
    if len(signal_one) > len(signal_two):
        signal_one = signal_one[::2]

    spectrum_one = np.abs(fft(signal_one))
    spectrum_two = np.abs(fft(signal_two))
    # reduce frequencies above 20000 hz
    spectrum_one = spectrum_one[:20000]
    spectrum_two = spectrum_two[:20000]

    manhattan_distance = np.sum(np.abs(spectrum_one - spectrum_two))
    # format manhattan_distance as scientific notation, with 3 decimal places
    manhattan_distance_sn = "{:.3e}".format(manhattan_distance)
    print(f'Manhattan distance of {manhattan_distance_sn} between frequencies of {file_one} and {file_two}: ')
    return manhattan_distance

def normalize_amplitudes(file_one, file_two):
    signal_one = get_signal(file_one)
    signal_two = get_signal(file_two)

    # normalize signal_two to signal_one
    print(f'ratios: file 1: {np.average(signal_one)}, file 2: {np.average(signal_two)}')
    signal_two = signal_two * np.average(signal_one) / np.average(signal_two)
    #save signal_two to new file
    # Convert the normalized signal back to bytes
    signal_two_bytes = signal_two.astype(np.int16).tobytes()

    wav_file_one = wave.open(file_one, 'r') # open file one to get parameters
    # remove .wav from file_two string
    file_two = file_two[:-4] 
    print(file_two)
    # Create a new wave file
    
    with wave.open(f'{file_two}_norm.wav', 'w') as wav_file:
        # Use the same parameters as the original file
        wav_file.setparams(wav_file_one.getparams())
        # Write the normalized signal to the new file
        wav_file.writeframes(signal_two_bytes)
    return signal_one, signal_two

def min_max_freq(freq_one, freq_two):
    min_both = min(min(freq_one), min(freq_two))
    max_both = max(max(freq_one), max(freq_two)) 
    # optional to trim freqs
    max_both = min(max_both, 20000) # remove frequencies above 20kHz which humans cant hear
    min_both = max(20, min_both) # remove frequencies below 20Hz which humans catn hear
    return min_both, max_both

def compare_amp_hist(file_one, file_two):
    # Read the corresponding frames from the file
    signal_one = get_signal(file_one)
    signal_two = get_signal(file_two)
    hist_diff = diff_histograms(signal_one, signal_two)
    hist_diff_sn = "{:.3e}".format(hist_diff)
    print(f'amplitude hist diff: {hist_diff_sn} between {file_one} and {file_two}')
    return hist_diff

def compare_freq_hist(file_one, file_two):
    signal_one = get_signal(file_one)
    signal_two = get_signal(file_two)
    # Get the frame rate
    framerate = get_framerate(file_one)

    freq_one, spectrum_one = compute_freq(signal_one, framerate)
    freq_two, spectrum_two = compute_freq(signal_two, framerate)

    min_both, max_both = min_max_freq(freq_one, freq_two)

    hist_1 = np.histogram(a=freq_one, bins=10, range=(min_both, max_both), weights=spectrum_one)
    hist_2 = np.histogram(a=freq_two, bins=10, range=(min_both, max_both), weights=spectrum_two)

    hist_diff = sum(np.abs(hist_1[0] - hist_2[0]))
    # format hist_diff as scientific notation, with 3 decimal places
    hist_diff_sn = "{:.3e}".format(hist_diff)
    print(f'frequency hist diff: {hist_diff_sn} between {file_one} and {file_two}')
    return hist_diff

def compare_db_hist(file_one, file_two):
    signal_one = get_signal(file_one)
    signal_two = get_signal(file_two)
    # Get the frame rate
    framerate = get_framerate(file_one)
    volume_db_one = 20 * np.log10(np.abs(signal_one)+10)
    volume_db_two = 20 * np.log10(np.abs(signal_two)+10)

    hist_diff = diff_histograms(volume_db_one, volume_db_two)
    hist_diff_sn = "{:.3e}".format(hist_diff)
    print(f'volume hist diff: {hist_diff_sn} between {file_one} and {file_two}')
    return hist_diff

def diff_histograms(hist_one, hist_two):
    min_both = min(min(hist_one), min(hist_two))
    max_both = max(max(hist_one), max(hist_two))
    hist_1 = np.histogram(a=hist_one, bins=10, range=(min_both, max_both))
    hist_2 = np.histogram(a=hist_two, bins=10, range=(min_both, max_both))
    hist_diff = sum(np.abs(hist_1[0] - hist_2[0]))
    return hist_diff

def moving_window_distance(file_one, file_two, window_duration_seconds=10):
    signal_one = get_signal(file_one)
    signal_two = get_signal(file_two)

    # process signal in window_duration_seconds windows
    window_size = window_duration_seconds * get_framerate(file_one)
    distance_amp, distance_freq = 0, 0
    for i in range(0, len(signal_one), window_size):
        window_one = signal_one[i:i+window_size]
        window_two = signal_two[i:i+window_size]
        distance_amp += diff_histograms(window_one, window_two)
        
        # compute frequency distance 
        framerate = get_framerate(file_one)
        
        freq_one, spectrum_one = compute_freq(window_one, framerate)
        freq_two, spectrum_two = compute_freq(window_two, framerate)

        min_both, max_both = min_max_freq(freq_one, freq_two)

        hist_1_freq = np.histogram(a=freq_one, bins=10, range=(min_both, max_both), weights=spectrum_one)
        hist_2_freq = np.histogram(a=freq_two, bins=10, range=(min_both, max_both), weights=spectrum_two)
        distance_freq = sum(np.abs(hist_1_freq[0] - hist_2_freq[0]))

        # compute volume distance
        volume_db_one = 20 * np.log10(np.abs(window_one)+10)
        volume_db_two = 20 * np.log10(np.abs(window_two)+10)
        distance_volume = diff_histograms(volume_db_one, volume_db_two)

    distance_amp_sn = "{:.3e}".format(distance_amp)
    distance_freq_sn = "{:.3e}".format(distance_freq)
    distance_volume_sn = "{:.3e}".format(distance_volume)
    print(f'Moving window distance of amplitude: {distance_amp_sn}, frequency: {distance_freq_sn}, volume: {distance_volume_sn} between {file_one} and {file_two}')
    return distance_amp, distance_freq, distance_volume_sn
