import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
import scipy.signal

def clean_audio(input_path: str, output_path: str):
    # Load audio
    y, sr = librosa.load(input_path, sr=16000, mono=True)

    # Noise reduction
    y_denoised = nr.reduce_noise(y=y, sr=sr)

    # High-pass filter (remove low-frequency rumble)
    sos = scipy.signal.butter(10, 100, 'hp', fs=sr, output='sos')
    y_filtered = scipy.signal.sosfilt(sos, y_denoised)

    # Normalize audio
    y_normalized = y_filtered / np.max(np.abs(y_filtered))

    # Save cleaned file
    sf.write(output_path, y_normalized, sr)
