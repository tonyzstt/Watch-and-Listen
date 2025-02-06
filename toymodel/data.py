import os
import torchaudio
import json
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

def pad_or_truncate(waveform, max_len):

    _, num_samples = waveform.shape
    
    if num_samples > max_len:
        waveform = waveform[:, :max_len]
    elif num_samples < max_len:
        pad_amount = max_len - num_samples
        waveform = F.pad(waveform, (0, pad_amount))
        
    return waveform

class VideoDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        with open(os.path.join(root_dir, "extracted_sequences.json"), "r") as f:
            self.data = json.load(f)

        self.audio_sample_rate = 16000

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        video_id = self.data[idx]
        json_path = os.path.join(self.root_dir, "videos", video_id, "metadata.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        text = data['text']

        image_dir = os.path.join(self.root_dir, "images", video_id)
        image_files = sorted(os.listdir(image_dir))
        total_frames = len(image_files)

        sampled_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
        sampled_files = [image_files[i] for i in sampled_indices]
        images = [Image.open(os.path.join(image_dir, img)) for img in sampled_files]
        if self.transform:
            images = [self.transform(img) for img in images]
        images = torch.stack(images) 
        audio_path = os.path.join(self.root_dir, "audios", f"{video_id}.wav")
        waveform, _ = torchaudio.load(audio_path)  
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True) 

        waveform = pad_or_truncate(waveform, max_len=441000)
        return images, waveform, text