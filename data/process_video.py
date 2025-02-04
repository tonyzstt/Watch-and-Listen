import os
import json
import cv2
import subprocess
from pathlib import Path
from tqdm import tqdm

# Define paths
IMAGE_OUTPUT_FOLDER = "images"
AUDIO_OUTPUT_FOLDER = "audios"
SUCCESS_JSON_PATH = "extracted_sequences.json"

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")

def process_videos(base_folder):

    image_out_path = os.path.join(base_folder, IMAGE_OUTPUT_FOLDER)
    audio_out_path = os.path.join(base_folder, AUDIO_OUTPUT_FOLDER)
    json_out_path = os.path.join(base_folder, SUCCESS_JSON_PATH)
    video_path = os.path.join(base_folder, "videos")
    os.makedirs(image_out_path, exist_ok=True)
    os.makedirs(audio_out_path, exist_ok=True)

    successful_sequences = []

    for subfolder in tqdm(os.listdir(video_path)):
        subfolder_path = os.path.join(video_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  

        metadata_path = os.path.join(subfolder_path, "metadata.json")

        if not os.path.exists(metadata_path):
            print(f"Skipping {subfolder}: metadata.json not found")
            continue

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        video_id = metadata["video_id"]
        start_time = metadata["start"]
        end_time = metadata["end"]

        video_file = None
        for file in os.listdir(subfolder_path):
            if file.startswith("video") and file.endswith(VIDEO_EXTENSIONS) and ".part" not in file:
                video_file = os.path.join(subfolder_path, file)
                break

        if not video_file:
            continue

        image_output_subfolder = os.path.join(image_out_path, video_id)
        audio_output_path = os.path.join(audio_out_path, f"{video_id}.wav")

        os.makedirs(image_output_subfolder, exist_ok=True)

        cap = cv2.VideoCapture(video_file)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if not ret or current_frame > end_frame:
                break

            frame_filename = os.path.join(image_output_subfolder, f"{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()

        audio_extracted = False
        try:
            subprocess.run([
                "ffmpeg", "-i", video_file, "-ss", str(start_time), "-to", str(end_time),
                "-q:a", "0", "-map", "a", audio_output_path, "-y"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            audio_extracted = True
        except subprocess.CalledProcessError:
            print(f"Error extracting audio for {video_id}")

        if frame_count > 0 and audio_extracted:
            successful_sequences.append(video_id)

    with open(json_out_path, "w") as json_file:
        json.dump(successful_sequences, json_file, indent=4)

    print("Processing complete!")

if __name__ == "__main__":
    process_videos("train")
    process_videos("test")