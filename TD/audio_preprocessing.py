import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

def apply_filters(audio_data, sample_rate, lowcut=100, highcut=8000):
    """Applies a band-pass filter to clean up the audio."""
    nyquist = 0.5 * sample_rate

    # Ensure valid cutoff frequencies
    low = max(0.0001, lowcut / nyquist)  # Ensure it's greater than zero
    high = min(0.9999, highcut / nyquist)  # Ensure it's less than Nyquist limit

    if low >= high:
        raise ValueError("Low cutoff frequency must be lower than high cutoff frequency.")

    b, a = butter(1, [low, high], btype="band")
    return filtfilt(b, a, audio_data)

def normalize_audio(audio_data):
    """Normalize the audio to prevent clipping and maintain consistency."""
    return audio_data / np.max(np.abs(audio_data))

def process_audio(audio_file):
    """Loads and processes audio, saving a cleaned version."""
    output_wav = "filtered_audio.wav"

    print(f"🔄 Processing audio: {audio_file}...")

    # Load audio using `soundfile`
    audio_data, sample_rate = sf.read(audio_file)

    # Ensure mono (if stereo, take mean of both channels)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Normalize audio
    audio_data = normalize_audio(audio_data)

    # Apply band-pass filtering
    filtered_audio = apply_filters(audio_data, sample_rate)

    # Normalize again after filtering
    filtered_audio = normalize_audio(filtered_audio)

    # Save processed audio as WAV using `soundfile`
    sf.write(output_wav, filtered_audio, sample_rate, subtype='PCM_16')

    print(f"✅ Audio processed and saved as '{output_wav}'.\n")
    return output_wav
