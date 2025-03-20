from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio import Model
from pydub import AudioSegment
import torch
import yaml
import os

# Ensure offline mode to prevent any downloads
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Define local paths for PyAnnote models
model_path = "/Users/rebecaleyva/Desktop/Postpartum/pyannote_models"

# ✅ Load pipeline configuration from `config.yaml`
config_path = f"{model_path}/speaker-diarization-3.1/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# ✅ Properly initialize the SpeakerDiarization pipeline using the loaded configuration
pipeline = SpeakerDiarization(**config)

# ✅ Manually load segmentation and embedding models from local storage
pipeline.segmentation = Model.from_pretrained(f"{model_path}/segmentation-3.0")
pipeline.embedding = Model.from_pretrained(f"{model_path}/embedding")

# ✅ Move the pipeline to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

# ✅ Define input audio file (must be .m4a)
input_audio = "audio.m4a"

# ✅ Convert .m4a to .wav (PyAnnote requires .wav format)
output_wav = "converted_audio.wav"
audio = AudioSegment.from_file(input_audio, format="m4a")
audio.export(output_wav, format="wav")

# ✅ Run the diarization pipeline on the converted audio file
diarization = pipeline(output_wav)

# ✅ Save the diarization output in RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)

print("Diarization complete! Results saved to 'audio.rttm'.")
