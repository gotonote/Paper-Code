"""
CASA: CNN Autoencoder with Score Attention
"""

import torch
import torch.nn as nn

class ScoreAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, 3, padding=1)
        self.score = nn.Conv2d(channel, 1, 1)
        
    def forward(self, x):
        feat = self.conv(x)
        score = torch.sigmoid(self.score(feat))
        return x * score

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256]):
        super().__init__()
        layers = []
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Conv2d(in_channels if i == 0 else hidden_dims[i-1], h, 4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(h))
            layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)

class ConvDecoder(nn.Module):
    def __init__(self, hidden_dims=[256, 128, 64], out_channels=3):
        super().__init__()
        layers = []
        for i, h in enumerate(hidden_dims):
            out_ch = hidden_dims[i+1] if i < len(hidden_dims)-1 else out_channels
            layers.append(nn.ConvTranspose2d(h, out_ch, 4, stride=2, padding=1))
            if i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm2d(hidden_dims[i+1]))
                layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.decoder(x)

class CASAModel(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256], latent_dim=128):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, hidden_dims)
        self.decoder = ConvDecoder(hidden_dims[::-1], in_channels)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.score_attention = nn.ModuleList([ScoreAttention(h) for h in hidden_dims])
        
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_var(h)
    
    def decode(self, z):
        h = z.view(z.size(0), -1, 1, 1)
        h = h.repeat(1, 1, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def create_casa_model(config):
    return CASAModel(
        in_channels=config.get('in_channels', 3),
        hidden_dims=config.get('hidden_dims', [64, 128, 256]),
        latent_dim=config.get('latent_dim', 128)
    )
