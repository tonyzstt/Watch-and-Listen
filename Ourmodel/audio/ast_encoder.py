import torch
import torch.nn as nn
from transformers import ASTFeatureExtractor, ASTModel  
import torchaudio
import torchaudio.transforms as T


class CLIPAudioFeatureTower(nn.Module):
    def __init__(self, audio_tower="MIT/ast-finetuned-audioset-10-10-0.4593", args=None, delay_load=False):
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
            self.cfg_only = None 

    def load_model(self):
        self.audio_processor = ASTFeatureExtractor.from_pretrained(self.audio_tower_name)
        self.audio_tower = ASTModel.from_pretrained(self.audio_tower_name)
        self.audio_tower.requires_grad_(False)  
        self.is_loaded = True

    def feature_select(self, audio_forward_outs):
        """
        Select features from the specified transformer layer.
        """
        audio_features = audio_forward_outs.hidden_states[self.select_layer]
        return audio_features

    def preprocess_audio(self, waveform, sample_rate):
        """
        Converts raw waveform into a spectrogram required for AST.
        """
        transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )
        mel_spectrogram = transform(waveform)
        return mel_spectrogram

    @torch.no_grad()
    def forward(self, audio_waveforms, sample_rate=16000):
        """
        Process audio waveforms and extract general audio features.
        """
        if isinstance(audio_waveforms, list): 
            audio_features = []
            for waveform in audio_waveforms:
                mel_spec = self.preprocess_audio(waveform, sample_rate)
                inputs = self.audio_processor(mel_spec, return_tensors="pt")
                inputs = {k: v.to(device=self.device, dtype=self.dtype) for k, v in inputs.items()}

                audio_forward_out = self.audio_tower(**inputs, output_hidden_states=True)
                audio_feature = self.feature_select(audio_forward_out).to(waveform.dtype)
                audio_features.append(audio_feature)
        else:
            mel_spec = self.preprocess_audio(audio_waveforms, sample_rate)
            inputs = self.audio_processor(mel_spec, return_tensors="pt")
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
        return self.audio_tower.config if self.is_loaded else None

    @property
    def hidden_size(self):
        return self.audio_tower.config.hidden_size
