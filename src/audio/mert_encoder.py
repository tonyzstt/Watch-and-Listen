import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModel


class MERTEncoder(nn.Module):
    def __init__(self, model_name: str = "m-a-p/MERT-v1-95M", processor_name: str = "m-a-p/MERT-v1-95M", sampling_rate: int = None):
        # Initialize the processor and model
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(processor_name, trust_remote_code=True)
        self.sampling_rate = self.processor.sampling_rate  # Default to processor's sampling rate
        if sampling_rate:
            self.sampling_rate = sampling_rate
    
    def load_audio(self, audio_file_name: str):
        # Load the audio file from the given path
        waveform, sampling_rate = torchaudio.load(audio_file_name)
        return waveform, sampling_rate
    
    def resample_audio(self, waveform: torch.Tensor, original_sampling_rate: int):
        # Resample the audio if necessary
        if original_sampling_rate != self.sampling_rate:
            # print(f'Resampling from {original_sampling_rate} to {self.sampling_rate}')
            resampler = T.Resample(original_sampling_rate, self.sampling_rate)
            waveform = resampler(waveform)
        return waveform
    
    def process_audio(self, audio_file_name: str):
        # Load and process the audio file
        waveform, sampling_rate = self.load_audio(audio_file_name)
        
        # Resample if necessary
        waveform = self.resample_audio(waveform, sampling_rate)
        
        # Process the waveform using the processor
        inputs = self.processor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt")
        return inputs, waveform
    
    def get_hidden_states(self, audio_file_name: str):
        # Get the model's hidden states (features) for a given audio file
        inputs, waveform = self.process_audio(audio_file_name)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract hidden states
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        
        # Time-reduced hidden states (mean pooling over time)
        time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
        
        # Apply weighted average using a learnable layer (optional)
        aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
        weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
        
        return weighted_avg_hidden_states