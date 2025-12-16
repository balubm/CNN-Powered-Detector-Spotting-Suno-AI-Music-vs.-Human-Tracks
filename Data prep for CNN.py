import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import random
import time
import os

#Step 1: Data Loading and Balanced Sampling
meta_data = pd.read_csv("/Users/balamuralibalu/PythonProjects/CLAP_AI_music_detection/Out/audio_files_metadata.csv")
suno_files = meta_data[meta_data['source'] == 'Suno'].sample(n=2000, random_state=42)
human_files = meta_data[meta_data['source'] == 'Human'].sample(n=2000, random_state=42)
selected_meta = pd.concat([suno_files, human_files]).sample(frac=1, random_state=42).reset_index(drop=True)
pd.DataFrame.to_csv(selected_meta, "/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/AI_music_detection/out/selected_audio_metadata.csv", index=False)
print(f"Total selected files: {len(selected_meta[selected_meta['source']=='Suno'])} Suno, {len(selected_meta[selected_meta['source']=='Human'])} Human")

BATCH_SIZE = 1000
data_for_ml = []
batch_num = 1
total_count = 0
short =0

# Step 2: Pre-check Human File Durations
for index, row in selected_meta.iterrows():
    if row["source"] == 'Human': #This is done only for Human tracks because Suno tracks are all more than 20 seconds
        y,sr = librosa.load(row['filepath'], sr = 44100)
        samples_needed = sr*20
        if len(y) < samples_needed:
            short +=1
            print (f"{short}/{index} is short")
print (f"{short} files are shorter than 20 seconds. Stop the process if you want to handle them differently.")
print("Starting audio processing and feature extraction...")

# Step 3: Audio Processing and Feature Extraction
for index, row in selected_meta.iterrows():
    y, sr = librosa.load(row["filepath"], sr=44100)
    samples_needed = sr * 20
    
    if len(y) > samples_needed:
        max_start = len(y) - samples_needed
        sample_start = random.randint(0, max_start)
        y = y[sample_start:sample_start+samples_needed]
    elif len(y) < samples_needed:
        print(f"File {row['filepath']},{row['source']}  is shorter than 20 seconds. Skipping.")
        continue
    # Step 4: Feature Extraction (STFT Spectrogram)
    s_pow = np.abs(librosa.stft(y, n_fft=4096, hop_length=512)) ** 2
    s_db = librosa.power_to_db(s_pow, ref=np.max)
    label = 1 if row["source"] == "Suno" else 0
    data_for_ml.append((s_db, label))
    total_count += 1
    # Step 5: Batch Saving
    if total_count % BATCH_SIZE == 0:
        spectrograms = np.array([d[0] for d in data_for_ml])
        labels = np.array([d[1] for d in data_for_ml])
        save_path = f'/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/AI_music_detection/out/data_batch_{batch_num}.npz'
        np.savez(save_path, spectrograms=spectrograms, labels=labels)
        print(f"Saved batch {batch_num} with {len(spectrograms)} items.")
        print(f"Processed {total_count} files so far.")
        batch_num += 1
        data_for_ml = []

# Step 6: Save any remainder
if data_for_ml:
    spectrograms = np.array([d[0] for d in data_for_ml])
    labels = np.array([d[1] for d in data_for_ml])
    save_path = f'/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/AI_music_detection/out/data_batch_{batch_num}.npz'
    np.savez(save_path, spectrograms=spectrograms, labels=labels)
    print(f"Saved final batch {batch_num} with {len(spectrograms)} items.")

    
# There will be total 4 batches of 1000 each since we have selected 4000 audio files. each batch will randomly have half Suno and half Human audio files approximately.