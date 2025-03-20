import os
import torch
import json
import numpy as np
import sys
from scipy.signal import convolve
from pydub import AudioSegment
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from scipy.io import wavfile

# -------------------------------
# Redirigir errores a un archivo log
# -------------------------------
log_file = "log.txt"
sys.stderr = open(log_file, "w")  # Redirige errores y advertencias a un archivo

# -------------------------------
# Configuración
# -------------------------------
audio_file = "audio.m4a"
output_wav = "converted_audio.wav"
reverb_wav = "reverb_audio.wav"
output_json_file = "transcription_with_speakers.json"

whisper_model_path = "/Users/rebecaleyva/Desktop/Postpartum/whisper-large-v3"
HUGGINGFACE_TOKEN = ""

# -------------------------------
# Convertir Audio a WAV (Requerido para procesamiento)
# -------------------------------
audio = AudioSegment.from_file(audio_file, format="m4a")
audio.export(output_wav, format="wav")

# -------------------------------
# Aplicar Reverb (Convolución)
# -------------------------------
sample_rate, audio_data = wavfile.read(output_wav)

# Convertir audio a mono si es necesario
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)

# Normalizar para evitar saturación
audio_data = audio_data / np.max(np.abs(audio_data))

# Filtro de reverb
reverb_filter = np.array([1, 0, 0, -1])

# Aplicar convolución
reverb_audio = convolve(audio_data, reverb_filter, mode="same")

# Normalizar el resultado
reverb_audio = reverb_audio / np.max(np.abs(reverb_audio))

# Guardar el audio con reverb
wavfile.write(reverb_wav, sample_rate, (reverb_audio * 32767).astype(np.int16))

print("✅ Reverb aplicada. Guardado como 'reverb_audio.wav'.")

# -------------------------------
# Diarización con PyAnnote
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGINGFACE_TOKEN
)
diarization_pipeline.to(device)

diarization = diarization_pipeline(reverb_wav)

speaker_segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    speaker_segments.append({
        "start_time": turn.start,
        "end_time": turn.end,
        "speaker": speaker,
        "transcript": ""
    })

print("✅ Diarización completada.")

# -------------------------------
# Transcripción con Whisper
# -------------------------------
processor = AutoProcessor.from_pretrained(whisper_model_path)
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
)
whisper_model.to("cpu")

whisper_transcriber = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    return_timestamps=True,
    device="cpu"
)

whisper_result = whisper_transcriber(reverb_wav, generate_kwargs={"language": "<|es|>"})
word_timestamps = whisper_result.get("chunks", [])

# -------------------------------
# Imprimir Transcripción Completa de Whisper
# -------------------------------
print("\n🔵 **Transcripción completa de Whisper:**")
for word in word_timestamps:
    print(f"[{word['timestamp'][0]:.2f}s - {word['timestamp'][1]:.2f}s] {word['text']}")

# -------------------------------
# Fusionar Diarización y Transcripción
# -------------------------------
for segment in speaker_segments:
    start_time = segment["start_time"]
    end_time = segment["end_time"]

    spoken_words = [
        word["text"] for word in word_timestamps
        if (start_time <= word["timestamp"][0] <= end_time) or
           (start_time <= word["timestamp"][1] <= end_time) or
           (word["timestamp"][0] <= start_time and word["timestamp"][1] >= end_time)
    ]

    segment["transcript"] = " ".join(spoken_words)

# -------------------------------
# Imprimir y Guardar JSON Final
# -------------------------------
print("\n🔵 **Diarización con Transcripción:**")
for segment in speaker_segments:
    print(f"{segment['speaker']} [{segment['start_time']:.2f}s - {segment['end_time']:.2f}s]: {segment['transcript']}")

with open(output_json_file, "w", encoding="utf-8") as json_file:
    json.dump(speaker_segments, json_file, ensure_ascii=False, indent=4)

print(f"✅ Transcripción con diarización guardada en '{output_json_file}'.")

# Cerrar archivo de errores
sys.stderr.close()
sys.stderr = sys.__stderr__  # Restaurar salida de errores a consola normal
