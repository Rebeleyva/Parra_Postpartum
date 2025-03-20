import json

# -------------------------------
# Load Diarization and Transcription Data
# -------------------------------
diarization_file = "diarization_results.json"  # Replace with your diarization JSON file
transcription_file = "whisper_transcription.json"  # Replace with your transcription JSON file
output_json_file = "aligned_transcription.json"

# Load diarization results
with open(diarization_file, "r", encoding="utf-8") as file:
    diarization_segments = json.load(file)

# Load transcription results
with open(transcription_file, "r", encoding="utf-8") as file:
    word_timestamps = json.load(file)

# -------------------------------
# Initialize transcription for each speaker segment
# -------------------------------
for segment in diarization_segments:
    segment["transcript"] = ""

# -------------------------------
# Assign Transcriptions Using Average Timestamp (and Find Nearest If Needed)
# -------------------------------
for word in word_timestamps:
    word_start, word_end = word["timestamp"]
    word_text = word["text"]
    
    # Calculate the average timestamp for this word
    average_time = (word_start + word_end) / 2

    # Try to find the correct diarization segment
    assigned = False
    closest_segment = None
    min_difference = float("inf")

    for segment in diarization_segments:
        start_time, end_time = segment["start_time"], segment["end_time"]

        # Check if the average timestamp falls inside this segment
        if start_time <= average_time <= end_time:
            if segment["transcript"]:
                segment["transcript"] += " " + word_text
            else:
                segment["transcript"] = word_text
            assigned = True
            break  # Stop once assigned

        # If not assigned, find the closest diarization segment
        difference = min(abs(start_time - average_time), abs(end_time - average_time))
        if difference < min_difference:
            min_difference = difference
            closest_segment = segment

    # If no exact match was found, assign to the nearest segment
    if not assigned and closest_segment:
        if closest_segment["transcript"]:
            closest_segment["transcript"] += " " + word_text
        else:
            closest_segment["transcript"] = word_text

# -------------------------------
# Save the Aligned Transcription JSON
# -------------------------------
with open(output_json_file, "w", encoding="utf-8") as json_file:
    json.dump(diarization_segments, json_file, ensure_ascii=False, indent=4)

print(f"✅ Aligned transcription saved in '{output_json_file}'.")

# -------------------------------
# Print Output for Verification
# -------------------------------
print("\n🔵 **Aligned Transcription:**")
for segment in diarization_segments:
    print(f"{segment['speaker']} [{segment['start_time']:.2f}s - {segment['end_time']:.2f}s]: {segment['transcript']}")
