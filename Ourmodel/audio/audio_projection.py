import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_audio_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    
    if projector_type == 'linear':
        return nn.Linear(config.audio_hidden_size, config.hidden_size)

    print("projector_type:", projector_type)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.audio_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    resnet_match = re.match(r'^resnet(\d+)?$', projector_type)
    if resnet_match:
        num_blocks = resnet_match.group(1)
        num_blocks = int(num_blocks) if num_blocks is not None else 2
        layers = [nn.Linear(config.audio_hidden_size, config.hidden_size)]
        for _ in range(num_blocks):
            layers.append(SimpleResBlock(config.hidden_size))
        return nn.Sequential(*layers)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
