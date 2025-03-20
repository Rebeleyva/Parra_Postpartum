import torch
import json
import numpy as np
import soundfile as sf
from io import BytesIO
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from scipy.io.wavfile import write

# Whisper model path
WHISPER_MODEL_PATH = r"C:\Users\luken\Desktop\ReLuVoice\ReLuVoiceAI\whisper-large-v3"

def transcribe_audio(audio_path, diarization_results):
    """Transcribes segmented audio using Whisper and aligns it with diarization results."""
    print("🔄 Running transcription with Whisper...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Whisper model
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(WHISPER_MODEL_PATH)
    whisper_transcriber = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
        batch_size=4,  # Batch processing for speed
        device=0 if torch.cuda.is_available() else -1
    )

    # Load processed audio using soundfile
    audio_data, sample_rate = sf.read(audio_path)

    # Ensure mono audio
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Extract audio segments for each speaker
    segment_arrays = []
    for segment in diarization_results:
        start_sample = int(segment["start_time"] * sample_rate)
        end_sample = int(segment["end_time"] * sample_rate)
        segment_audio = audio_data[start_sample:end_sample]
        segment_arrays.append(segment_audio)

    # Perform batched transcription (convert segments to NumPy arrays)
    transcriptions = []
    for segment_audio in segment_arrays:
        if len(segment_audio) == 0:
            transcriptions.append({"text": ""})  # Handle empty segments
            continue

        transcription = whisper_transcriber(
            {"sampling_rate": sample_rate, "raw": segment_audio},
            generate_kwargs={"language": "<|es|>"},
        )
        transcriptions.append(transcription)

    # Attach transcripts to diarization results
    for i, segment in enumerate(diarization_results):
        segment["transcript"] = transcriptions[i]["text"]

    # Save transcription results
    with open("aligned_transcription.json", "w", encoding="utf-8") as json_file:
        json.dump(diarization_results, json_file, ensure_ascii=False, indent=4)

    print("✅ Transcription completed. Results saved in 'aligned_transcription.json'.\n")
    return diarization_results
