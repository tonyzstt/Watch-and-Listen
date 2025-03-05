import librosa
import numpy as np

class AudioTrainProcessor():

    def __init__(self,
                 sample_rate=22050,
                 sample_length=4.0,
                 mono=False,
                 min_scale=-32768.0,
                 max_scale=32767.0):
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        self.mono = mono
        self.min_scale = min_scale
        self.max_scale = max_scale

    def scale(self, old_value, old_min, old_max, new_min, new_max):
        if old_max == old_min:
            return np.full_like(old_value, (new_min + new_max) / 2, dtype=np.float32)

        old_range = (old_max - old_min)
        new_range = (new_max - new_min)
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min
        return new_value

    def preprocess(self, waveform, sr):

        waveform = waveform.T  

        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        # Convert between mono/stereo
        if waveform.shape[0] == 1 and not self.mono:
            waveform = np.concatenate([waveform, waveform], axis=0)
        elif waveform.shape[0] > 1 and self.mono:
            waveform = np.mean(waveform, axis=0, keepdims=True)

        # Truncate to sample_length seconds
        # TODO: maybe we should pad and mask later
        max_samples = int(sr * self.sample_length)
        waveform = waveform[:, :max_samples]

        # Scale the amplitude
        old_min, old_max = waveform.min(), waveform.max()
        waveform = self.scale(waveform, old_min, old_max, self.min_scale, self.max_scale)

        return {'audio_wav': waveform.astype(np.float32)}

    