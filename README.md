# Speaker Diarization and Recognition Pipeline

## **Overview**
This repository provides a Python-based pipeline for speaker diarization and identification. The tool leverages state-of-the-art pre-trained models for speaker embedding extraction and diarization, enabling accurate segmentation and recognition of speakers in audio recordings.

---

## **Features**
- Speaker Diarization: Automatically segments audio files into speaker-specific segments.

- Speaker Identification: Matches unknown speakers in the audio file with a set of pre-recorded voices.

- Custom Thresholds: Configurable similarity threshold for speaker matching.

- Visualization: Generates visual annotations of speaker segments.

- Noise Reduction: Outputs cleaner individual speaker audio clips. 

## **Setup and Installation**

### **1. Requirements**
- Python 3.8 or higher.
- Libraries:
    ```bash
    pip install torch torchaudio speechbrain pyannote.audio scipy matplotlib numpy librosa soundfile
    ```

### **2. Hugging Face Token**
Obtain a token from [Hugging Face](https://huggingface.co/settings/tokens) and include it in the script for using `pyannote.audio`.

---

## **Folder Structure**
Below is the required folder structure:

 

## **Usage**

### **1. Input Files**
- **Multi-Speaker Audio**: Place the file (e.g., `meeting.wav`) in `speech_data/input/multiple_people_wav/`.
- **Known Speakers Audio**: Place files (e.g., `John.wav`, `David.wav`) in `speech_data/input/individual_wav/`.

### **2. Run the Script**
Execute the script:
```bash
python pyannote-audio.py
