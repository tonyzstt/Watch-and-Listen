import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2Config


class AudioEncoder(nn.Module):
    def __init__(self, audio_tower, args=None, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.audio_tower_name = audio_tower

        if args is None:
            self.select_layer = -2  
        else:
            self.select_layer = args.audio_select_layer

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = Wav2Vec2Config.from_pretrained(self.audio_tower_name)

    def load_model(self):
        self.audio_processor = Wav2Vec2Processor.from_pretrained(self.audio_tower_name)
        self.audio_tower = Wav2Vec2Model.from_pretrained(self.audio_tower_name)
        self.audio_tower.requires_grad_(False) 
        self.is_loaded = True

    def feature_select(self, audio_forward_outs):
        """
        Select features from the specified transformer layer.
        """
        audio_features = audio_forward_outs.hidden_states[self.select_layer]  
        return audio_features

    @torch.no_grad()
    def forward(self, audio_waveforms):
        """
        Process audio waveforms and extract features.
        """
        if isinstance(audio_waveforms, list):  
            audio_features = []
            for waveform in audio_waveforms:
                inputs = self.audio_processor(waveform, return_tensors="pt", sampling_rate=16000)
                inputs = {k: v.to(device=self.device, dtype=self.dtype) for k, v in inputs.items()}
                
                audio_forward_out = self.audio_tower(**inputs, output_hidden_states=True)
                audio_feature = self.feature_select(audio_forward_out).to(waveform.dtype)
                audio_features.append(audio_feature)
        else:
            inputs = self.audio_processor(audio_waveforms, return_tensors="pt", sampling_rate=16000)
            inputs = {k: v.to(device=self.device, dtype=self.dtype) for k, v in inputs.items()}
            
            audio_forward_outs = self.audio_tower(**inputs, output_hidden_states=True)
            audio_features = self.feature_select(audio_forward_outs).to(audio_waveforms.dtype)

        return audio_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
