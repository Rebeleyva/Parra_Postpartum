import torch
import json
from pyannote.audio import Pipeline

HUGGINGFACE_TOKEN = ""  # Add your token if required

def perform_diarization(audio_path):
    """Performs speaker diarization and returns structured JSON results."""
    print("🔄 Running speaker diarization...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_TOKEN
    )
    diarization_pipeline.to(device)

    diarization = diarization_pipeline(audio_path)
    diarization_results = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_results.append({
            "start_time": turn.start,
            "end_time": turn.end,
            "speaker": speaker
        })

    with open("diarization_results.json", "w", encoding="utf-8") as json_file:
        json.dump(diarization_results, json_file, ensure_ascii=False, indent=4)

    print("✅ Diarization completed. Results saved in 'diarization_results.json'.\n")
    return diarization_results
