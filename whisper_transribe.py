import os
import torch
import json
import sys
import tkinter as tk
from tkinter import filedialog
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# -------------------------------
# Select audio file with a GUI
# -------------------------------
root = tk.Tk()
root.withdraw()  # Hide Tkinter main window
audio_file = filedialog.askopenfilename(
    title="Select an audio file",
    filetypes=[("Audio files", "*.m4a *.wav *.mp3 *.flac")]
)
if not audio_file:
    print("❌ No file selected.")
    sys.exit(1)

print(f"✅ Selected file: {audio_file}")

# -------------------------------
# Define Whisper Model Path
# -------------------------------
model_path = "/Users/rebecaleyva/Desktop/Postpartum/whisper-large-v3"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
)

# Move model to CPU
model.to("cpu")

# Set up the pipeline (explicitly set to CPU)
transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    return_timestamps=True,  # Enable timestamps
    device="cpu"
)

# -------------------------------
# Transcribe the Audio
# -------------------------------
result = transcriber(audio_file, generate_kwargs={"language": "<|es|>"})
word_timestamps = result.get("chunks", [])

print("✅ Transcription complete!")

# -------------------------------
# Print Each Word with Its Timestamp
# -------------------------------
print("\n🔵 **Transcription with Timestamps:**")
for word in word_timestamps:
    start_time = word["timestamp"][0]
    end_time = word["timestamp"][1]
    text = word["text"]
    print(f"[{start_time:.2f}s - {end_time:.2f}s] {text}")

# -------------------------------
# Save JSON with Timestamps
# -------------------------------
output_json_file = "whisper_transcription.json"
with open(output_json_file, "w", encoding="utf-8") as json_file:
    json.dump(word_timestamps, json_file, ensure_ascii=False, indent=4)

print(f"✅ Transcription saved in '{output_json_file}'.")
