import torch.nn as nn
import timm

class VisionEncoder(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", embed_dim=768):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.proj = nn.Linear(self.encoder.num_features, embed_dim)

    def forward(self, images):
        """
        images: [batch_size, num_frames, C, H, W]
        We'll flatten frames into the batch dimension for the encoder, then un-flatten.
        """
        b, f, c, h, w = images.shape
        images = images.view(b * f, c, h, w)

        feats = self.encoder(images)            
        feats = self.proj(feats)             
        feats = feats.view(b, f, -1)        
        return feats