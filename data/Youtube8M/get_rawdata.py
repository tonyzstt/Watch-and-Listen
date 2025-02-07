import os
import pandas as pd
import yt_dlp
import json
from tqdm import tqdm

def download_video(video_id, save_dir):
    """Download the video using yt_dlp."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    video_path = os.path.join(save_dir, "video.mp4")

    ydl_opts = {
        "format": "bestvideo+bestaudio/best", 
        "outtmpl": video_path,
        "quiet": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            return video_path
    except Exception as e:
        print(f"Error downloading video {video_id}: {e}")
        return None

def save_metadata(video_id, text, start, end, save_dir):
    metadata = {
        "video_id": video_id,
        "start": start,
        "end": end,
        "text": text
    }

    metadata_path = os.path.join(save_dir, "metadata.json")

    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        return metadata_path
    except Exception as e:
        print(f"Error saving metadata for {save_dir}: {e}")
        return None

def get_dataset(csv_file, folder):

    os.makedirs(folder, exist_ok=True)
    df = pd.read_csv(csv_file)

    for _, row in tqdm(df.iterrows()):

        video_id = row["video_id"]
        start = row["start"]
        end = row["end"]
        text = row["text"]
        video_dir = os.path.join(folder, video_id)
        os.makedirs(video_dir, exist_ok=True)
        
        if os.path.exists(os.path.join(video_dir, "metadata.json")):
            print(f"Skipping video {video_id} as it already exists.")
            continue

        download_video(video_id, video_dir)
        save_metadata(video_id, text, start, end, video_dir)

if __name__ == "__main__":
    get_dataset("train.csv", "train/videos")
    get_dataset("test.csv", "test/videos")
