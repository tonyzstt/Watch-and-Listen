import os
import json
import cv2
import subprocess
from tqdm import tqdm

PROCESSED_DATA_ROOT = "processed_data"
IMAGE_OUTPUT_FOLDER = "images"
AUDIO_OUTPUT_FOLDER = "audios"
META_OUTPUT_FOLDER = "metas"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")

def process_videos(input_folder, split_name):

    output_folder = os.path.join(PROCESSED_DATA_ROOT, split_name)
    os.makedirs(output_folder, exist_ok=True)

    image_out_path = os.path.join(output_folder, IMAGE_OUTPUT_FOLDER)
    audio_out_path = os.path.join(output_folder, AUDIO_OUTPUT_FOLDER)
    meta_out_path = os.path.join(output_folder, META_OUTPUT_FOLDER)
    os.makedirs(image_out_path, exist_ok=True)
    os.makedirs(audio_out_path, exist_ok=True)
    os.makedirs(meta_out_path, exist_ok=True)

    video_input_folder = input_folder

    successful_sequences = []

    if not os.path.isdir(video_input_folder):
        print(f"Warning: {video_input_folder} does not exist or is not a directory.")
        return successful_sequences

    for subfolder in tqdm(os.listdir(video_input_folder), desc=f"Processing {split_name}"):
        subfolder_path = os.path.join(video_input_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        
        video_id = subfolder_path.split('/')[-1]
        metadata_path = os.path.join(subfolder_path, f"{video_id}.json")

        if not os.path.exists(metadata_path):
            print(f"Skipping {subfolder}: metadata.json not found.")
            continue

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        video_id = metadata.get("video_id", subfolder)
        start_frame = metadata.get("start", 0)
        start_frame = metadata.get("clip_start_end_idx", 0)[0]
        end_frame = metadata.get("clip_start_end_idx", 0)[1]

        video_file = None
        for file_name in os.listdir(subfolder_path):
            if file_name.startswith("video") and file_name.endswith(VIDEO_EXTENSIONS) and ".part" not in file_name:
                video_file = os.path.join(subfolder_path, file_name)
                break
        
        if not video_file:
            print(f"No valid video file found for {video_id}. Skipping...")
            continue

        image_output_subfolder = os.path.join(image_out_path, video_id)
        audio_output_file = os.path.join(audio_out_path, f"{video_id}.wav")
        os.makedirs(image_output_subfolder, exist_ok=True)

        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_time = float(start_frame / fps)
        end_time = float(end_frame / fps)
        if fps <= 0:
            print(f"Warning: FPS is 0 or invalid for {video_id}. Skipping...")
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Extract images
        frame_count = 0
        while cap.isOpened():
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if current_pos > end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = os.path.join(image_output_subfolder, f"{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()

        # Extract audio
        audio_extracted = False
        if frame_count > 0:
            try:
                subprocess.run([
                    "ffmpeg", "-i", video_file,
                    "-ss", str(start_time),
                    "-to", str(end_time),
                    "-q:a", "0",
                    "-map", "a",
                    audio_output_file, 
                    "-y"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True)
                audio_extracted = True
            except subprocess.CalledProcessError:
                print(f"Error extracting audio for {video_id}")

        if frame_count > 0 and audio_extracted:
            
            # Append ID and save metadata
            successful_sequences.append(video_id)
            updated_metadata = {
                "original_metadata": metadata,  
                "video_id": video_id,
                "start_time": start_time,
                "end_time": end_time,
                "fps": fps,
                "frame_count": frame_count,
                "images_folder": os.path.relpath(image_output_subfolder, start=output_folder),
                "audio_file": os.path.relpath(audio_output_file, start=output_folder),
            }

            # Write JSON
            json_out_path = os.path.join(meta_out_path, f"{video_id}.json")
            with open(json_out_path, "w", encoding="utf-8") as jf:
                json.dump(updated_metadata, jf, indent=4)

        else:
            print(f"Skipping {video_id}: no frames extracted or audio failed.")

    return successful_sequences

def main():

    splits = ["train", "val", "test"]
    all_valid_ids = {}

    for split_name in splits:
        input_folder = os.path.join("video_dataset", split_name)
        valid_ids = process_videos(input_folder, split_name)
        all_valid_ids[split_name] = valid_ids

    for split_name, video_ids in all_valid_ids.items():
        out_file = os.path.join(PROCESSED_DATA_ROOT, f"{split_name}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(video_ids, f, indent=4)

    print("All splits processed. Done!")

if __name__ == "__main__":
    main()
