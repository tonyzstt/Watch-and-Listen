import torch
import torch.nn as nn

from audio.wav2vec_encoder import AudioEncoder
from vision.clip_encoder import CLIPVisionTower
from vision.vision_projection import build_vision_projector
from audio.audio_projection import build_audio_projector
from config.config import VisionProjectorConfig, AudioProjectorConfig
from dataloader import _get_rawvideo_dec

class VideoSummaryModel(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        audio_model_name="facebook/wav2vec2-base-960h",
        vision_config_params=None,
        audio_config_params=None
    ):
        super(VideoSummaryModel, self).__init__()
        if vision_config_params is None:
            vision_config_params = {
                "mm_projector_type": "mlp2x_gelu",
                "mm_hidden_size": 768,
                "hidden_size": 512
            }
        if audio_config_params is None:
            audio_config_params = {
                "mm_projector_type": "mlp2x_gelu",
                "audio_hidden_size": 768,
                "hidden_size": 512
            }
        
        self.clip_vision_tower = CLIPVisionTower(clip_model_name)
        self.audio_encoder = AudioEncoder(audio_model_name)
        self.vision_projector = build_vision_projector(
            VisionProjectorConfig(**vision_config_params)
        )
        self.audio_projector = build_audio_projector(
            AudioProjectorConfig(**audio_config_params)
        )
    
    def forward(self, patch_images, audio_data):
        """
        Expects:
            patch_images: List or batch of preprocessed video frame images.
            audio_data: Preprocessed audio data.
        Returns:
            A tuple (vision_features, audio_feature) where:
              - vision_features is the output from the vision branch.
              - audio_feature is the output from the audio branch.
        """
        image_features = self.clip_vision_tower(patch_images)
        image_features = torch.cat(image_features, dim=0)
        vision_features = self.vision_projector(image_features)
    
        audio_feature = self.audio_encoder(audio_data)
        audio_feature = self.audio_projector(audio_feature)
        
        return vision_features, audio_feature

if __name__ == "__main__":

    video_path = "/home/tonyzst/Desktop/CS229-Project/data/MMTrail/test/videos/2x2NMwBDzE.mp4"
    audio_path = "/home/tonyzst/Desktop/CS229-Project/data/MMTrail/test/audios/-2x2NMwBDzE.wav"
    
    patch_images, slice_len, video_mask, audio_rate, audio_data = _get_rawvideo_dec(
        video_path, audio_path=audio_path
    )
    
    print("Number of frames:", slice_len)
    print("Audio sample rate:", audio_rate)
    
    model = VideoSummaryModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    patch_images = [img.to(device) for img in patch_images]
    audio_data = audio_data.to(device)
    
    vision_features, audio_feature = model(patch_images, audio_data)
    
    if isinstance(vision_features, list) and len(vision_features) > 0:
        print("Feature shape for the first frame:", vision_features[0].shape)
    else:
        print("Video Feature shape:", vision_features.shape)
        print("Audio Feature shape:", audio_feature.shape)
