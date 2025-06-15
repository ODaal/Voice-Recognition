import sounddevice as sd
import wavio
import os

def record_audio(filename, duration, fs):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    wavio.write(filename, recording, fs, sampwidth=2)

if __name__ == "__main__":
    user_name = input("Enter your name: ")
    duration = 5  # Duration of the recording in seconds
    fs = 16000  # Sample rate

    # Create the directory if it doesn't exist
    if not os.path.exists('recordings'):
        os.makedirs('recordings')

    filename = f"recordings/{user_name}.wav"
    record_audio(filename, duration, fs)