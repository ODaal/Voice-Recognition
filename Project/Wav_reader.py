import librosa
import numpy as np
from scipy.signal import lfilter, butter
import VGGpreprocess
import Constants as c



def read_audio(filename, sample_rate):
	audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
	audio = audio.flatten()
	return audio


def load_wav(filename, sample_rate):
	audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
	audio = audio.flatten()
	return audio

	############################## I am not the Owner of this code #########################################

def normalize_frames(m,epsilon=1e-12):
	return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])


def remove_dc_and_dither(sin, sample_rate):
	if sample_rate == 16e3:
		alpha = 0.99
	elif sample_rate == 8e3:
		alpha = 0.999
	else:
		print("Sample rate must be 16kHz or 8kHz only")
		exit(1)
	sin = lfilter([1,-1], [1,-alpha], sin)
	dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
	spow = np.std(dither)
	sout = sin + 1e-6 * spow * dither
	return sout


def get_fft_spectrum(filename, buckets):
	signal = load_wav(filename,c.SAMPLE_RATE)
	signal *= 2**15

	# get FFT spectrum
	signal = remove_dc_and_dither(signal, c.SAMPLE_RATE)
	signal = VGGpreprocess.preemphasis(signal, coeff=c.PREEMPHASIS_ALPHA)
	frames = VGGpreprocess.framesig(signal, frame_len=c.FRAME_LEN*c.SAMPLE_RATE, frame_step=c.FRAME_STEP*c.SAMPLE_RATE, winfunc=np.hamming)
	fft = abs(np.fft.fft(frames,n=c.NUM_FFT))
	fft_norm = normalize_frames(fft.T)

	# truncate to max bucket sizes
	rsize = max(k for k in buckets if k <= fft_norm.shape[1])
	rstart = int((fft_norm.shape[1]-rsize)/2)
	out = fft_norm[:,rstart:rstart+rsize]

	return out

def read_and_process_audio(filename, buckets):
	signal = read_audio(filename,c.SAMPLE_RATE)

	# # Filter out non-speech frequencies
	# lowcut, highcut = c.FILTER_RANGE
	# signal = butter_bandpass_filter(signal, lowcut, highcut, c.SAMPLE_RATE, 1)

	# # Normalize signal
	# signal = normalize(signal)

	signal *= 2**15

	# Process signal to get FFT spectrum
	signal = remove_dc_and_dither(signal, c.SAMPLE_RATE)
	signal = VGGpreprocess.preemphasis(signal, coeff=c.PREEMPHASIS_ALPHA)
	frames = VGGpreprocess.framesig(signal, frame_len=c.FRAME_LEN*c.SAMPLE_RATE, frame_step=c.FRAME_STEP*c.SAMPLE_RATE, winfunc=np.hamming)
	fft = abs(np.fft.fft(frames,n=c.NUM_FFT))
	fft_norm = normalize_frames(fft.T)

	# Truncate to middle MAX_SEC seconds
	rsize = max(k for k in buckets if k <= fft_norm.shape[1])
	rstart = int((fft_norm.shape[1]-rsize)/2)
	out = fft_norm[:,rstart:rstart+rsize]

	return out
