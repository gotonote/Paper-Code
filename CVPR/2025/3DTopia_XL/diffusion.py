import torch
import torch.nn as nn

class PrimitiveDiffusion(nn.Module):
    """Primitive扩散模型"""
    
    def __init__(self, num_primitives=256):
        super().__init__()
        self.num_primitives = num_primitives
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4),
            num_layers=6
        )
        self.decoder = nn.Linear(128, 10)  # position(3) + scale(3) + rotation(4)
        
    def forward(self, text_features, noisy_primitives):
        x = self.encoder(noisy_primitives + text_features)
        return self.decoder(x)
