from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
import os

# Your Hugging Face token (replace with your actual token)
HUGGINGFACE_TOKEN = ""

# Load the model using the token
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGINGFACE_TOKEN
)

# Move pipeline to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)


# Define input audio file (must be .m4a)
input_audio = "audio.m4a"

# Convert .m4a to .wav (PyAnnote requires .wav)
output_wav = "converted_audio.wav"
audio = AudioSegment.from_file(input_audio, format="m4a")
audio.export(output_wav, format="wav")

# Run the diarization pipeline on the converted audio file
diarization = pipeline(output_wav)

# Save the diarization output in RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)

print("Diarization complete! Results saved to 'audio.rttm'.")

