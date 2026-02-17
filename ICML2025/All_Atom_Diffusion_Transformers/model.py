"""
All-Atom Diffusion Transformers
全原子扩散变换器
"""

import torch
import torch.nn as nn

class AtomDiffusion(nn.Module):
    def __init__(self, num_atoms=1000, hidden_dim=256):
        super().__init__()
        
        self.atom_encoder = nn.Embedding(num_atoms, hidden_dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=6
        )
        
        self.coord_head = nn.Linear(hidden_dim, 3)  # xyz坐标
        self.vel_head = nn.Linear(hidden_dim, 3)   # 速度
        
    def forward(self, atom_types, positions):
        emb = self.atom_encoder(atom_types)
        features = self.transformer(emb + positions)
        
        coords = self.coord_head(features)
        velocities = self.vel_head(features)
        
        return coords, velocities
