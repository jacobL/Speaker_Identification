Speaker Diarization and Recognition Pipeline
Overview
This project is a Python-based pipeline for:

Speaker diarization: Identifying segments of speech from different speakers in a meeting audio file.
Speaker recognition: Associating identified segments with known speakers based on pre-provided voice samples.
The project uses state-of-the-art models for speaker embedding extraction and diarization, integrating tools like SpeechBrain, Pyannote Audio, and others.

Features
Perform speaker diarization on multi-speaker audio files.
Visualize diarization results as a timeline plot.
Match diarized segments with known speakers using speaker embeddings.
Save speaker-specific audio segments as separate files.
Setup and Requirements
1. Prerequisites
Python 3.8 or higher.
A machine with CUDA support (optional but recommended for faster processing).
2. Required Libraries
Install dependencies with:

bash
複製
編輯
pip install torch torchaudio speechbrain pyannote.audio scipy matplotlib numpy librosa soundfile
3. Hugging Face API Token
To use the Pyannote Audio pipeline, you need a valid Hugging Face API token. Obtain it from Hugging Face.

Usage
1. Input File Structure
Multi-speaker audio file: Place it in speech_data/input/multiple_people_wav/ (e.g., Trump_Fridman_2m.wav).
Known speakers' audio files: Place in speech_data/input/individual_wav/. Use filenames as speaker IDs (e.g., David.wav, John.wav).
2. Running the Script
Run the main script to perform diarization and recognition:

bash
複製
編輯
python main.py
3. Outputs
Speaker diarization results:
Text file: speech_data/results/speaker_diarization_txt/
Timeline plot: speech_data/results/segments/
Matched segments with known speakers:
RTTM file: speech_data/results/segments/
Speaker-specific audio clips:
WAV files: speech_data/results/individual_remove_background_noise_wav/
Code Details
Key Functions
list_files_with_full_path(directory_path)
Lists all files in a given directory with their full paths.

plot_and_save_segments_wide(segments, output_path)
Visualizes and saves diarization segments as a timeline plot.

get_timestamp()
Generates a timestamp string for uniquely naming output files.

Diarization Pipeline
Uses the Pyannote Audio pipeline to identify speaker segments.

Speaker Recognition
Extracts speaker embeddings using SpeechBrain and matches segments with known speakers based on cosine similarity.

Example Workflow
Place the multi-speaker audio file (Trump_Fridman_2m.wav) in the input folder.
Add known speakers' audio samples (David.wav, John.wav) in the respective folder.
Run the script.
Check the output folders for:
Diarization results.
Matched speaker segments.
Individual speaker audio files.
Notes
Adjust the similarity threshold in the script (threshold = 0.8) for better speaker matching accuracy.
Ensure high-quality, noise-free audio files for better diarization and recognition results.
References
SpeechBrain
Pyannote Audio
Librosa Documentation
