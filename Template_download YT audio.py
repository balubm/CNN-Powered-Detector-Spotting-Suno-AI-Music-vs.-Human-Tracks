import yt_dlp
import pandas as pd
import os

# Load the MusicCaps CSV metadata file
musiccaps = pd.read_csv('/Users/balamuralibalu/PythonProjects/AI_Music_detection_project/AI_music_detection/musiccaps-public.csv')  # Adjust file name/path as necessary



# Directory to save audio clips
os.makedirs("musiccaps_audio", exist_ok=True)

for idx, row in musiccaps.iterrows():
    # Get YouTube id and clip seconds
    yt_id = row["ytid"]
    start = int(row["start_s"])
    end = int(row["end_s"])
    duration = end - start
    filename = f"musiccaps_audio/{yt_id}_{start}_{end}.wav"
    
    # Download 10s audio segment
    if not os.path.exists(filename):
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': filename,
            'ffmpeg_location': '/Users/balamuralibalu/bin/ffmpeg',     # Top-level only!
            'ffprobe_location': '/Users/balamuralibalu/bin/ffprobe',   # Top-level only!
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'download_sections': [f"*{start}-{end}"],
        }

        url = f"https://www.youtube.com/watch?v={yt_id}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    print(f"Downloaded {filename}")

print("Bulk download complete!")
