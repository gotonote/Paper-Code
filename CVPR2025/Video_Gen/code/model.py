"""
Video Generation 模型实现

本实现为第三方复现，仅供参考
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResBlock3D(nn.Module):
    """3D ResNet Block"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)
    
    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class VideoVAE(nn.Module):
    """3D Video VAE for video encoding/decoding"""
    
    def __init__(self, in_channels=3, latent_channels=4):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            ResBlock3D(64),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            ResBlock3D(128),
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            ResBlock3D(256),
            nn.Conv3d(256, latent_channels * 2, 3, padding=1)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(latent_channels, 256, 3, padding=1),
            ResBlock3D(256),
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            ResBlock3D(128),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            ResBlock3D(64),
            nn.Conv3d(64, in_channels, 3, padding=1)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        return z
    
    def decode(self, z):
        return self.decoder(z)


class SpatiotemporalAttention(nn.Module):
    """Spatiotemporal Attention for video"""
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias for spatial
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, (8*8)))
    
    def forward(self, x, T):
        """x: [B*T, N, C]"""
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        
        # Reshape for spatiotemporal processing
        x = x.view(B // T, T, H, W, C)
        x = x.reshape(B // T, T * H * W, C)
        
        # Multi-head attention
        qkv = self.qkv(x).reshape(B // T, T * H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B // T, T * H * W, C)
        return self.proj(x)


class VideoDiffusionModel(nn.Module):
    """Video Diffusion Model with spatiotemporal attention"""
    
    def __init__(self, vae, latent_dim=4, hidden_dim=256, num_heads=8, num_layers=12):
        super().__init__()
        self.vae = vae
        
        # Input projection
        self.input_proj = nn.Conv3d(latent_dim, hidden_dim, 3, padding=1)
        
        # Transformer blocks with spatiotemporal attention
        self.blocks = nn.ModuleList([
            TransformerBlock3D(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Conv3d(hidden_dim, latent_dim, 3, padding=1)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x, t):
        """Generate video frames"""
        # Encode
        z = self.vae.encode(x)
        
        # Time embedding
        t_emb = self.get_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        
        # Process through transformer
        h = self.input_proj(z)
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Decode
        out = self.output_proj(h)
        return self.vae.decode(out)
    
    @staticmethod
    def get_timestep_embedding(timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class TransformerBlock3D(nn.Module):
    """3D Transformer Block with spatiotemporal attention"""
    
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SpatiotemporalAttention(hidden_dim, num_heads)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x, t_emb):
        B, C, T, H, W = x.shape
        
        # Reshape to sequence
        x = rearrange(x, 'b c t h w -> (t h w) b c').transpose(0, 1)
        
        # Add time embedding
        x = x + t_emb.unsqueeze(1)
        
        # Attention
        x = x + self.attn(self.norm1(x), T)
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        # Reshape back
        x = x.transpose(0, 1)
        x = rearrange(x, '(t h w) b c -> b c t h w', t=T, h=H, w=W)
        
        return x


def create_video_gen_model():
    """Create video generation model"""
    vae = VideoVAE(in_channels=3, latent_channels=4)
    model = VideoDiffusionModel(vae, latent_dim=4, hidden_dim=256)
    return model
