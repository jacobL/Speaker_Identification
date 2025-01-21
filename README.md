# Speaker Diarization and Recognition Pipeline

## **Overview**
This pipeline performs:
1. **Speaker Diarization**: Identifying segments of speech from different speakers in a multi-speaker audio file.
2. **Speaker Recognition**: Matching diarized segments with known speakers using pre-recorded audio samples.

---

## **Features**
- Segment multi-speaker audio into speaker-specific timestamps.
- Visualize speaker diarization results as a timeline plot.
- Match diarized segments with known speakers based on voice embeddings.
- Save speaker-specific audio segments as separate WAV files.

---

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
