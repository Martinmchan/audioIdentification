import numpy as np
import resampy, os
from scipy.io import wavfile
from scipy.fftpack import dct

def wav_to_mfcc(wav_file):
	SAMPLE_RATE = 16000
	WINDOW_SIZE = 0.04*SAMPLE_RATE
	sr, data = wavfile.read(wav_file)
	data = data[:,0]
	data = resampy.resample(data, sr, SAMPLE_RATE)

	pre_emp_ratio = 0.97
	emp_data = np.append(data[0], data[1:] - pre_emp_ratio * data[:-1])


	FRAME_SIZE = 0.025
	FRAME_STRIDE = 0.01

	frame_length, frame_step = FRAME_SIZE * SAMPLE_RATE, FRAME_STRIDE * SAMPLE_RATE  # Convert from seconds to samples
	signal_length = len(emp_data)
	frame_length = int(round(frame_length))
	frame_step = int(round(frame_step))
	num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

	pad_signal_length = num_frames * frame_step + frame_length
	z = np.zeros((pad_signal_length - signal_length))
	pad_signal = np.append(emp_data, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

	indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
	frames = pad_signal[indices.astype(np.int32, copy=False)]

	frames *= np.hamming(frame_length) #Apply Hamming window

	NFFT = 512
	mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
	pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum


	nfilt = 40
	low_freq_mel = 0
	high_freq_mel = (2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700))  # Convert Hz to Mel
	mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
	hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
	bin = np.floor((NFFT + 1) * hz_points / SAMPLE_RATE)

	fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
	for m in range(1, nfilt + 1):
		f_m_minus = int(bin[m - 1])   # left
		f_m = int(bin[m])             # center
		f_m_plus = int(bin[m + 1])    # right

	for k in range(f_m_minus, f_m):
		fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
	for k in range(f_m, f_m_plus):
		fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
	filter_banks = np.dot(pow_frames, fbank.T)
	filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
	filter_banks = 20 * np.log10(filter_banks)  # dB

	num_ceps = 12
	mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')#[:, 1 : (num_ceps + 1)] # Keep 2-13



path = "/home/martinch/Documents/audioIdentification/gun_shot_8K"
os.chdir(path)
files = os.listdir(".")
print(wav_to_mfcc(files[0]))