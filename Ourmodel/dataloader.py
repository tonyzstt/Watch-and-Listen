from PIL import Image
from decord import VideoReader, cpu
import numpy as np
import os
import torch
import torchaudio
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def image_preprocess(image, image_size=224):
    """
    Preprocess a PIL image using default normalization values from BaseProcessor.

    Args:
        image (PIL.Image): The input image.
        image_size (int, optional): The size to which the image will be resized (both height and width). Defaults to 224.

    Returns:
        dict: A dictionary with key 'pixel_values' containing a list with the processed tensor.
    """
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    pixel_values = transform(image)
    return {'pixel_values': [pixel_values]}


def load_audio(audio_path, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(audio_path)
    
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze(0)
    waveform = waveform.to(torch.float32)
    
    return waveform, sample_rate

def _get_rawvideo_dec(video_path, max_frames=64, image_resolution=224, video_framerate=1, s=None, e=None, audio_path=None):
    """
    Extracts video frames using decord and loads audio from a separate WAV file.

    Args:
        video_path (str): Path to the video file.
        image_processor (callable): A function that preprocesses a PIL image.
        max_frames (int): Maximum number of video frames to process.
        image_resolution (int): The resolution for processed images.
        video_framerate (int): Frame rate for sampling video frames.
        s (optional): Start time for the video segment.
        e (optional): End time for the video segment.
        audio_path (str, optional): Path to the WAV audio file.

    Returns:
        tuple: (patch_images, slice_len, video_mask, audio_rate, audio_data)
    """
    video_mask = np.zeros(max_frames, dtype=np.int64)
    patch_images = []  

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        raise FileNotFoundError(f"Video file not found: {video_path}")

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1

    if num_frames > 0:
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))
        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[i] for i in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        else:
            sample_pos = all_pos

        raw_frames = vreader.get_batch(sample_pos).asnumpy()
        patch_images = [Image.fromarray(f) for f in raw_frames]
        patch_images = [image_preprocess(img)['pixel_values'][0] for img in patch_images]
        slice_len = len(patch_images)

        video_mask[:slice_len] = 1

        while len(patch_images) < max_frames:
            patch_images.append(torch.zeros((3, image_resolution, image_resolution)))
    else:
        print(f"Video path: {video_path} error.")
        slice_len = 0

    if audio_path is None:
        raise ValueError("An audio_path must be provided to load audio from a WAV file.")
    audio_data, audio_rate = load_audio(audio_path)

    return patch_images, slice_len, video_mask, audio_rate, audio_data


if __name__ == '__main__':
    video_path = "/home/tonyzst/Desktop/CS229-Project/data/MMTrail/test/videos/2x2NMwBDzE.mp4"
    audio_path = "/home/tonyzst/Desktop/CS229-Project/data/MMTrail/test/audios/-2x2NMwBDzE.wav"
    patch_images, slice_len, video_mask, audio_rate, audio_data = _get_rawvideo_dec(
        video_path, audio_path=audio_path
    )
    print("Number of frames:", slice_len)
    print("Audio sample rate:", audio_rate)
