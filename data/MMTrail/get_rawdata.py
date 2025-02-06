import os
import json
import yt_dlp
from tqdm import tqdm

def download_video(video_id, save_dir):
    """
    Download the video using yt_dlp.
    Returns the path to the downloaded video (or None if download fails).
    """
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

def get_dataset(json_file, root_folder="video_dataset"):
    """
    Read a JSON file with video data, filter videos by duration (< 500s),
    and split them into 3 subsets: train/val/test in an 8:2:2 ratio.
    Each valid (short) video is downloaded and placed in its subset folder,
    and the corresponding JSON metadata is also saved in that folder.
    """

    # 1. Load data from JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Ensure the root_folder exists
    os.makedirs(root_folder, exist_ok=True)

    # Keep track of how many short videos have been processed to determine the subset
    short_video_count = 0

    # 2. Iterate over each video item in the JSON
    for item in tqdm(data, desc="Processing videos"):
        # Check duration (skip if >= 500 seconds)
        video_duration = item.get("video_duration", 0)
        if video_duration >= 500:
            continue

        # Determine subset based on short_video_count % 12
        subset_index = short_video_count % 12
        if subset_index < 8:
            subset_name = "train"
        elif subset_index < 10:
            subset_name = "val"
        else:
            subset_name = "test"

        short_video_count += 1

        # Pull out the video_id; if missing, skip
        video_id = item.get("video_id", None)
        if not video_id:
            continue
        
        # Create the subset folder, e.g. video_dataset/train/zwqH6_yWVEI
        subset_folder = os.path.join(root_folder, subset_name)
        video_folder = os.path.join(subset_folder, video_id)
        os.makedirs(video_folder, exist_ok=True)

        # 3. Download the video
        downloaded_path = download_video(video_id, video_folder)
        if not downloaded_path:
            # If download failed, you can decide to remove the folder or skip
            continue

        # 4. Save the *entire* video info (item) to a JSON file in that folder
        metadata_path = os.path.join(video_folder, f"{video_id}.json")
        try:
            with open(metadata_path, "w", encoding="utf-8") as meta_f:
                json.dump(item, meta_f, indent=4)
        except Exception as e:
            print(f"Error saving metadata for video {video_id}: {e}")

if __name__ == "__main__":
    # Example usage:
    # Adjust "data.json" and "video_dataset" to your actual file/folder names if needed
    get_dataset("data.json", "video_dataset")
