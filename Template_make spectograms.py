import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d
from pathlib import Path

import soundfile as sf

# --- CONFIG ---
suno_folder = "/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/AI_music_detection/suno_tracks/"
human_folder = "/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/AI_music_detection/human_tracks/"
#https://www.chosic.com/free-music/piano/
output_folder = "/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/AI_music_detection/spectrograms/"
os.makedirs(output_folder, exist_ok=True)

# --- FUNCTION: Generate Spectrogram ---
def generate_spectrogram(audio_path):
    # info = sf.info(audio_path)
    # print(info)
    # print("Bit depth:", info.subtype)
    y, sr = librosa.load(audio_path, sr=None)
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=512))**2
    S_db = librosa.power_to_db(S, ref=np.max)
    
    return S_db

# --- FUNCTION: Save Spectrogram Image ---
def save_spectrogram_image(S_db, save_path):
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, x_axis='time', y_axis='log')
    plt.ylim(0, 60)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(save_path)
    plt.close()

# --- LOAD TRACKS ---
def load_tracks(folder, name):

    print(f"folder is {folder}")
    tracks = []
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            S_db = generate_spectrogram(path)
            save_spectrogram_image(S_db, os.path.join(output_folder, f"{name}_{file}_spec.png"))
            tracks.append((file, S_db))
    return tracks


# --- FUNCTION: Compute Cross-Correlation ---
# def compute_correlation(S1, S2):
#     """
#     Compute normalized 2D cross-correlation between two spectrograms.
#     Returns a value between -1 and 1.
#     """
#     # Ensure both spectrograms are float
#     S1 = S1.astype(np.float32)
#     S2 = S2.astype(np.float32)
    
#     # Subtract mean and normalize each
#     S1_norm = (S1 - np.mean(S1)) / (np.std(S1) + 1e-8)
#     S2_norm = (S2 - np.mean(S2)) / (np.std(S2) + 1e-8)
    
#     # Compute 2D correlation
#     corr = correlate2d(S1_norm, S2_norm, mode='valid')
    
#     # Normalize by number of elements to get correlation coefficient
#     corr /= (S1_norm.shape[0] * S1_norm.shape[1])
    
#     # Return the maximum correlation value
#     return np.max(corr)


# --- ANALYZE CROSS-CORRELATIONS ---
# def analyze_folder(tracks, label):
#     print(f"\n--- Pairwise Correlations for {label} ---")
#     n = len(tracks)
#     print(f"Number of tracks: {n}")
#     for i in range(n):
#         for j in range(i+1, n):
#             print(f"Comparing Track {i+1} and Track {j+1}")
#             file1, S1 = tracks[i]
#             file2, S2 = tracks[j]
#             corr = compute_correlation(S1, S2)
#             print(f"{file1} <-> {file2}: {corr:.2f}")


def main():
    print("Generating spectrograms and saving images...")
    suno_tracks = load_tracks(suno_folder, "suno")
    human_tracks = load_tracks(human_folder,"human")
    # analyze_folder(suno_tracks, "Suno Tracks")
    # analyze_folder(human_tracks, "Human Tracks")
    print("Script finished!")
if __name__ == "__main__":
    main()