import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentEncoder(nn.Module):
    """内容编码器"""
    
    def __init__(self, in_channels=3, hidden_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        return self.encoder(x)


class StyleEncoder(nn.Module):
    """风格编码器"""
    
    def __init__(self, in_channels=3, style_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
        )
        
        self.style映射 = nn.Linear(256, style_dim)
        
    def forward(self, x):
        feat = self.encoder(x)
        feat = feat.flatten(2).transpose(1, 2)
        style = self.style映射(feat.mean(dim=1))
        return style


class StyleGenerator(nn.Module):
    """风格化生成器"""
    
    def __init__(self, latent_dim=512, style_dim=64):
        super().__init__()
        
        self.initial_conv = nn.Conv2d(latent_dim, 512, 3, 1, 1)
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.InstanceNorm2d(512),
                nn.ReLU()
            )
            for _ in range(6)
        ])
        
        self.style_inject = nn.ModuleList([
            StyleInjection(style_dim) for _ in range(6)
        ])
        
        self.to_rgb = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, content_feat, style_vec):
        x = self.initial_conv(content_feat)
        
        for layer, style_inj in zip(self.layers, self.style_inject):
            x = layer(x)
            x = style_inj(x, style_vec)
        
        output = self.to_rgb(x)
        return output


class StyleInjection(nn.Module):
    """风格注入模块"""
    
    def __init__(self, style_dim):
        super().__init__()
        self.style_scale = nn.Linear(style_dim, 512)
        self.style_bias = nn.Linear(style_dim, 512)
    
    def forward(self, x, style):
        scale = self.style_scale(style).unsqueeze(-1).unsqueeze(-1)
        bias = self.style_bias(style).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + bias


class StyleGANLlava(nn.Module):
    """StyleGAN-Llava 风格化图像生成"""
    
    def __init__(self, latent_dim=512, style_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        
        # 内容编码器
        self.content_encoder = ContentEncoder()
        
        # 风格编码器
        self.style_encoder = StyleEncoder(style_dim=style_dim)
        
        # 生成器
        self.generator = StyleGenerator(latent_dim, style_dim)
        
        # 映射网络
        self.mapping = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
    def encode_text(self, text):
        """简单的文本编码"""
        return torch.randn(text.shape[0], 512)
    
    def fuse_style_text(self, style_feat, text_feat):
        """融合风格和文本特征"""
        return style_feat + 0.3 * text_feat
    
    def forward(self, content_img, style_ref, text_prompt=None):
        # 编码内容
        content_feat = self.content_encoder(content_img)
        
        # 编码风格
        style_feat = self.style_encoder(style_ref)
        
        # 融合文本（可选）
        if text_prompt is not None:
            text_feat = self.encode_text(text_prompt)
            style_feat = self.fuse_style_text(style_feat, text_feat)
        
        # 映射到潜在空间
        latent = self.mapping(style_feat)
        
        # 生成
        output = self.generator(content_feat, latent)
        
        return output


if __name__ == "__main__":
    model = StyleGANLlava()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试
    content = torch.randn(2, 3, 256, 256)
    style = torch.randn(2, 3, 256, 256)
    output = model(content, style)
    print(f"Output shape: {output.shape}")
