import torch.nn as nn
import torch

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.attn_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, vision_features, audio_features):

        fused = torch.cat((vision_features, audio_features), dim=1)
        fused = fused.transpose(0, 1)  # (v_seq+a_seq, b, embed_dim)
        out = self.attn_layers(fused)  # (v_seq+a_seq, b, embed_dim)
        out = out.transpose(0, 1)      # (b, v_seq+a_seq, embed_dim)
        return out