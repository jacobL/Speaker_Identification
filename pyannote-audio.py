import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from pyannote.audio import Pipeline
from pyannote.core import notebook
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
import librosa
import soundfile as sf 

input_multiple_people = r"speech_data\\input\\multiple_people_wav\\Trump_Fridman.wav"
input_multiple_people = r"speech_data\\input\\multiple_people_wav\\Trump_Fridman_2m.wav"

def list_files_with_full_path(directory_path):
    try:
        # 列出目錄中的所有檔案和子目錄，並篩選出檔案
        file_path = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        file_path.sort()  # 按字母順序排序完整路徑
        return file_path
    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' does not exist.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory_path}'.")
        return []

def get_timestamp(): 
    return datetime.now().strftime('%Y%m%d%H%M%S')

def plot_and_save_segments_wide(segments, output_path="segments_wide.png"):
    fig, ax = plt.subplots(figsize=(15, 1))  # Increase figure size for better spacing

    # Plot the segments
    notebook.plot_annotation(segments, ax=ax, time=True, legend=True)
    
    # Customize title, labels, and spacing
    #ax.set_title("Speaker Diarization Segments", fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Speaker", fontsize=14)
    ax.tick_params(axis='x', labelsize=12)  # Increase x-axis label size
    ax.tick_params(axis='y', labelsize=12)  # Increase y-axis label size

    # Save the plot as a PNG file
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory
    print(f"Wide segments image saved as: {output_path}")

timestamp = get_timestamp()    
# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model for speaker embedding extraction and move it to the device
# Note: You need to obtain an API key from Hugging Face to use this model.
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
classifier = classifier.to(device)

diarization = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")

segments = diarization(input_multiple_people) 
with open("speech_data\\results\\segments\\segments_"+timestamp+".rttm", "w") as file:
    segments.write_rttm(file)

print("Segments saved to 'segments.rttm'")

# Call the function to save the wider segments plot
plot_and_save_segments_wide(segments, r"speech_data\\results\\segments\\segments_wide_"+timestamp+".png")

known_speakers = []
known_speaker_ids = []
# 示例使用
directory = "speech_data\input\individual_wav"  # 替換為目標資料夾路徑
file_path = list_files_with_full_path(directory)
for path in file_path:
    print(path)
    file_name = os.path.splitext(os.path.basename(path))[0]
    print(file_name)
    waveform, sample_rate = torchaudio.load(path )
    waveform = waveform.to(device)
    embedding = classifier.encode_batch(waveform)
    known_speakers.append(embedding.squeeze(1).cpu().numpy())   
    known_speaker_ids.append(file_name)

# Set a threshold for similarity scores to determine when a match is considered successful
threshold = 0.8

speaker_audio = {} 
signal, sr = librosa.load(input_multiple_people, sr=None)

# 将 librosa 加载的音频保存为临时 WAV 文件（因为 pyannote 需要音频文件输入）
temp_wav_path = "temp.wav"
sf.write(temp_wav_path, signal, sr)
speaker_diarization_output_file = r"speech_data\results\speaker_diarization_txt\speaker_diarization_output_"+timestamp+".txt"
with open(speaker_diarization_output_file, 'w', encoding='utf-8') as file:
    # Iterate through each segment identified in the diarization process
    for segment, label, confidence in segments.itertracks(yield_label=True):
        start_time, end_time = segment.start, segment.end

        # Load the specific audio segment from the meeting recording
        waveform, sample_rate = torchaudio.load(input_multiple_people, num_frames=int((end_time-start_time)*sample_rate), frame_offset=int(start_time*sample_rate))
        waveform = waveform.to(device)

        # Extract the speaker embedding from the audio segment
        embedding = classifier.encode_batch(waveform).squeeze(1).cpu().numpy()

        # Initialize variables to find the recognized speaker
        min_distance = float('inf')
        recognized_speaker_id = None

        # Compare the segment's embedding to each known speaker's embedding using cosine distance
        for i, speaker_embedding in enumerate(known_speakers): 
            distances = cdist(embedding, speaker_embedding, metric="cosine")
            min_distance_candidate = distances.min()
            if min_distance_candidate < min_distance:
                min_distance = min_distance_candidate
                recognized_speaker_id = known_speaker_ids[i]

                # 截取对应话者的音频片段
                if recognized_speaker_id not in speaker_audio:
                    speaker_audio[recognized_speaker_id] = []
                speaker_audio[recognized_speaker_id].append(signal[int(start_time * sr):int(end_time * sr)])

        # Output the identified speaker and the time range they were speaking, if a match is found
        if min_distance < threshold:
            file.write(f"Speaker {recognized_speaker_id} speaks from {start_time}s to {end_time}s.\n")
        else:
            file.write(f"No matching speaker found for segment from {start_time}s to {end_time}s.\n")

for speaker, audio_clips in speaker_audio.items():
    #combined_audio = librosa.util.fix_length(sum(audio_clips), len(signal))  # 合并音频片段
    combined_audio = np.concatenate(audio_clips)

    # 如果需要对齐到原始信号长度，可以使用 librosa.util.fix_length
    #combined_audio = librosa.util.fix_length(combined_audio, len(signal))
    output_path = f"speech_data\\results\\individual_remove_background_noise_wav\\speaker_{speaker}_"+timestamp+".wav"
    sf.write(output_path, combined_audio, sr)
    print(f"已保存话者 {speaker} 的音频到 {output_path}")
##########################ˇ


"""
signal, sr = librosa.load(input_multiple_people, sr=None)

# 将 librosa 加载的音频保存为临时 WAV 文件（因为 pyannote 需要音频文件输入）
temp_wav_path = "temp.wav"
sf.write(temp_wav_path, signal, sr)

# 使用 pyannote 处理音频，进行话者分离
#segments = diarization(temp_wav_path)
 
# 可选：将不同话者的音频分离并保存
speaker_audio = {}  # 用于存储每个话者的音频数据
  
for segment, _, speaker in segments.itertracks(yield_label=True):
    print(f"Speaker {speaker}: start={segment.start:.1f}s, end={segment.end:.1f}s")
    start = int(segment.start * sr)
    end = int(segment.end * sr)
    
    # 截取对应话者的音频片段
    if speaker not in speaker_audio:
        speaker_audio[speaker] = []
    speaker_audio[speaker].append(signal[start:end])
for speaker, audio_clips in speaker_audio.items():
    #combined_audio = librosa.util.fix_length(sum(audio_clips), len(signal))  # 合并音频片段
    combined_audio = np.concatenate(audio_clips)

    # 如果需要对齐到原始信号长度，可以使用 librosa.util.fix_length
    #combined_audio = librosa.util.fix_length(combined_audio, len(signal))
    output_path = f"speaker_{speaker}.wav"
    sf.write(output_path, combined_audio, sr)
    print(f"已保存话者 {speaker} 的音频到 {output_path}")
"""
