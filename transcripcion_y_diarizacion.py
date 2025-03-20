import os
import torch
import json
import sys
import tkinter as tk
from tkinter import filedialog
from scipy.signal import butter, filtfilt
from pydub import AudioSegment, effects
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from scipy.io import wavfile
import numpy as np

def apply_filters(audio_data, sample_rate, lowcut=100, highcut=8000):
    """Applies a band-pass filter to clean up the audio."""
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype="band")
    filtered_audio = filtfilt(b, a, audio_data)
    return filtered_audio

root = tk.Tk()
root.withdraw()
audio_file = filedialog.askopenfilename(
    title="Select an audio file",
    filetypes=[("Audio files", "*.m4a *.wav *.mp3")]
)
if not audio_file:
    print("❌ No file selected.")
    sys.exit(1)

print(f"✅ Selected file: {audio_file}")

log_file = "log.txt"
sys.stderr = open(log_file, "w")

output_wav = "filtered_audio.wav"
whisper_json_file = "whisper_transcription.json"
diarization_json_file = "diarization_results.json"
aligned_json_file = "aligned_transcription.json"
diarization_rttm = "audio.rttm"

whisper_model_path = "/Users/rebecaleyva/Desktop/Postpartum/whisper-large-v3"
HUGGINGFACE_TOKEN = ""

audio = AudioSegment.from_file(audio_file)
audio = audio.set_channels(1)
audio = effects.normalize(audio)
audio = audio.low_pass_filter(8000).high_pass_filter(100)
audio.export(output_wav, format="wav")

sample_rate, audio_data = wavfile.read(output_wav)
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)
audio_data = audio_data / np.max(np.abs(audio_data))
filtered_audio = apply_filters(audio_data, sample_rate)
filtered_audio = filtered_audio / np.max(np.abs(filtered_audio))
wavfile.write(output_wav, sample_rate, (filtered_audio * 32767).astype(np.int16))

print("✅ Audio processed and saved as 'filtered_audio.wav'.")

# -------------------------------
# Perform Speaker Diarization with PyAnnote
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGINGFACE_TOKEN
)
diarization_pipeline.to(device)

diarization = diarization_pipeline(output_wav)

with open(diarization_rttm, "w") as rttm:
    diarization.write_rttm(rttm)

print(f"✅ Speaker diarization complete! Results saved in '{diarization_rttm}'.")

diarization_results = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    diarization_results.append({
        "start_time": turn.start,
        "end_time": turn.end,
        "speaker": speaker
    })

with open(diarization_json_file, "w", encoding="utf-8") as json_file:
    json.dump(diarization_results, json_file, ensure_ascii=False, indent=4)

print(f"✅ Diarization results saved in '{diarization_json_file}'.")

# -------------------------------
# Transcribe the Full Audio with Whisper using diarization timestamps
# -------------------------------
processor = AutoProcessor.from_pretrained(whisper_model_path)
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
)
whisper_model.to("cuda" if torch.cuda.is_available() else "cpu")

whisper_transcriber = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    return_timestamps=True,
    device=0 if torch.cuda.is_available() else -1
)

for segment in diarization_results:
    segment_audio = audio[segment["start_time"] * 1000 : segment["end_time"] * 1000]
    temp_segment_path = "temp_segment.wav"
    segment_audio.export(temp_segment_path, format="wav")
    
    transcription = whisper_transcriber(temp_segment_path, generate_kwargs={"language": "<|es|>"})
    segment["transcript"] = " ".join([chunk["text"] for chunk in transcription.get("chunks", [])])

# Save the Aligned Transcription JSON
with open(aligned_json_file, "w", encoding="utf-8") as json_file:
    json.dump(diarization_results, json_file, ensure_ascii=False, indent=4)

print(f"✅ Aligned transcription saved in '{aligned_json_file}'.")

sys.stderr.close()
sys.stderr = sys.__stderr__
