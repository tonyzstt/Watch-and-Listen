import torchaudio.transforms as T
import torch.nn as nn
import torch

class AudioEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.mfcc_transform = T.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64},
        )
        self.conv = nn.Sequential(
            nn.Conv1d(40, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, embed_dim, kernel_size=5, stride=2, padding=2),
        )

    def forward(self, waveform):

        b, _, _ = waveform.shape
        all_mfcc = []
        for i in range(b):
            single_wave = waveform[i]  # shape [channels, time]
            
            # shape [1, time]
            if single_wave.shape[0] > 1:
                single_wave = single_wave.mean(dim=0, keepdim=True)
            
            mfcc = self.mfcc_transform(single_wave) 
            if mfcc.dim() == 3 and mfcc.shape[0] == 1:
                mfcc = mfcc.squeeze(0)  # [40, time]
            
            all_mfcc.append(mfcc) 

        mfcc = torch.stack(all_mfcc, dim=0)
        conv_out = self.conv(mfcc)  
        conv_out = conv_out.permute(0, 2, 1)
        return conv_out
