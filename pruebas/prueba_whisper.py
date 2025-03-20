from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

# Define model path
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
    device="cpu"  # Ensure CPU usage
)

# Define audio file
audio_file = "audio.m4a"

# Transcribe audio (forcing Spanish)
result = transcriber(audio_file, generate_kwargs={"language": "<|es|>"})

# Print the transcription
print(result["text"])
