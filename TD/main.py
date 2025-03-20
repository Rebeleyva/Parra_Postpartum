import sys
import argparse
from audio_preprocessing import process_audio
from diarization import perform_diarization
from transcription import transcribe_audio

def main():
    parser = argparse.ArgumentParser(description="AI Audio Transcription & Diarization Pipeline")
    parser.add_argument("audio_file", type=str, help="Path to the input audio file")

    args = parser.parse_args()
    audio_file = args.audio_file

    print("\n🚀 Running AI Audio Processing Pipeline...\n")

    # Step 1: Audio Processing
    processed_audio = process_audio(audio_file)

    # Step 2: Speaker Diarization
    diarization_results = perform_diarization(processed_audio)

    # Step 3: Transcription
    transcribed_results = transcribe_audio(processed_audio, diarization_results)

    print("\n✅ Pipeline Completed! Transcriptions saved in 'aligned_transcription.json'.\n")

if __name__ == "__main__":
    main()
