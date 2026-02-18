"""
M4Bench: Multi-Image Understanding
多图像理解模型
"""

import torch
import torch.nn as nn
from transformers import BertModel

class MultiImageEncoder(nn.Module):
    def __init__(self, num_images=4, embed_dim=768):
        super().__init__()
        self.num_images = num_images
        self.image_proj = nn.Linear(512, embed_dim)
        
    def forward(self, images):
        # images: [B, N, C, H, W]
        B, N, C, H, W = images.shape
        images_flat = images.view(B * N, C, H, W)
        
        # 简化处理
        feat = images_flat.mean(dim=[2, 3])
        feat = self.image_proj(feat)
        
        return feat.view(B, N, -1)

class M4Model(nn.Module):
    def __init__(self, num_images=4, num_classes=100):
        super().__init__()
        self.image_encoder = MultiImageEncoder(num_images)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        self.fusion = nn.Linear(768 * 2, 768)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, images, text_ids, text_mask):
        img_feat = self.image_encoder(images)
        img_feat = img_feat.mean(dim=1)
        
        text_feat = self.text_encoder(text_ids, text_mask).last_hidden_state[:, 0]
        
        fused = torch.cat([img_feat, text_feat], dim=-1)
        fused = F.relu(self.fusion(fused))
        
        return self.classifier(fused)
