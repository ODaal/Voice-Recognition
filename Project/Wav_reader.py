import wave
import numpy as np
from scipy.signal import lfilter, butter
import VGGpreprocess
import Constants as c


def read_audio(filename, sample_rate):
    print(f"Wav_reader.read_audio: Attempting to load filename='{filename}', type='{type(filename)}', sample_rate='{sample_rate}'")
    audio, sr = librosa.load(path = "/storage/emulated/0/Android/data/com.example.voiceapp/cache/recorded_audio.wav", sr=16000, mono=True)
    audio = audio.flatten()
    return audio

## For mobilee app system integration
"""
def read_audio(filename, sample_rate):
	try:
		with wave.open(filename, 'rb') as wf:
			# --- Validation ---
			if wf.getnchannels() != 1:
				raise ValueError("Audio file must be mono.")
			if wf.getframerate() != sample_rate:
				raise ValueError(f"Unsupported sample rate: {wf.getframerate()}. "
								 f"Expected {sample_rate}.")
			if wf.getsampwidth() != 2:
				raise ValueError("Audio file must be 16-bit PCM.")

			# --- Read and Process Data ---
			num_frames = wf.getnframes()
			audio_bytes = wf.readframes(num_frames)

			# Convert the byte string into a NumPy array of 16-bit integers
			audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

			# Normalize the audio to the range [-1.0, 1.0] as librosa does.
			# 32768.0 is used because it's the max value for a 16-bit signed integer.
			audio_float = audio_int16.astype(np.float32) / 32768.0

			# The flatten() call is no longer needed as the output is already 1D.
			return audio_float

	except wave.Error as e:
		print(f"Error reading WAV file: {e}")
		return None
	except Exception as e:
		print(f"An unexpected error occurred in read_audio: {e}")
		return None
"""


def normalize_frames(frames,epsilon=1e-12):
	return np.array([(i - np.mean(i)) / max(np.std(i),epsilon) for i in frames]) # Usual Normalization stuff


def remove_dc_and_dither(sig, sample_rate):
	if sample_rate == 16e3:
		alpha = 0.99
	elif sample_rate == 8e3:
		alpha = 0.999
	else:
		print("Sample rate must be 16kHz or 8kHz only")
		exit(1)
	sig = lfilter([1,-1], [1,-alpha], sig) # Remove DC low pass filter
	dither=np.random.uniform(low=-1,high=1, size=sig.shape)
	rootpower = np.std(dither)
	sout = sig + 1e-6 * rootpower * dither # reduce quantization noise by dithering
	return sout


def get_fft_spectrum(filename, buckets):
	signal = read_audio(filename,c.SAMPLE_RATE)
	signal *= 2**15

	# get FFT spectrum
	signal = remove_dc_and_dither(signal, c.SAMPLE_RATE) # removing DC and distortions
	signal = VGGpreprocess.preemphasis(signal, coeff=c.PREEMPHASIS_ALPHA) # enhance high frequencies
	frames = VGGpreprocess.framesig(signal, frameLen=c.FRAME_LEN*c.SAMPLE_RATE, frameStep=c.FRAME_STEP*c.SAMPLE_RATE, winfunc=np.hamming) # framing the signal
	fft = abs(np.fft.fft(frames,n=c.NUM_FFT)) # Applyinng the fast fourier transform
	fft_norm = normalize_frames(fft.T) # Normalizing the data

	# truncate to max bucket sizes
	rsize = max(b for b in buckets if b <= fft_norm.shape[1]) # Signal fft size shouldn't be bigger than buckets
	fix = int((fft_norm.shape[1]-rsize)/2)
	out = fft_norm[:,fix:fix+rsize]

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
	signal = remove_dc_and_dither(signal, c.SAMPLE_RATE) # removing DC and distortions
	signal = VGGpreprocess.preemphasis(signal, coeff=c.PREEMPHASIS_ALPHA) # enhance high frequencies
	frames = VGGpreprocess.framesig(signal, frameLen=c.FRAME_LEN*c.SAMPLE_RATE, frameStep=c.FRAME_STEP*c.SAMPLE_RATE, winfunc=np.hamming)  # framing the signal
	fft = abs(np.fft.fft(frames,n=c.NUM_FFT)) # Applyinng the fast fourier transform
	fft_norm = normalize_frames(fft.T) # Normalizing the data

	# Truncate to middle MAX_SEC seconds
	rsize = max(b for b in buckets if b <= fft_norm.shape[1])
	fix = int((fft_norm.shape[1]-rsize)/2)
	out = fft_norm[:,fix:fix+rsize]

	return out


