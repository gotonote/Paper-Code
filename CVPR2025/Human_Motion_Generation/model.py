"""
Human Motion Generation 实现

本实现为第三方复现，仅供参考
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEncoder(nn.Module):
    """运动编码器"""
    
    def __init__(self, num_joints=22, hidden_dim=512):
        super().__init__()
        self.num_joints = num_joints
        
        # 关节位置编码
        self.joint_encoder = nn.Linear(num_joints * 3, hidden_dim)
        
        # 时间卷积
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        )
    
    def forward(self, motion):
        """
        Args:
            motion: [B, T, J, 3] 关节位置序列
        """
        B, T, J, C = motion.shape
        
        # 展平关节维度
        motion = motion.view(B, T, J * C)
        
        # 编码
        features = self.joint_encoder(motion)
        
        # 时间建模
        features = features.transpose(1, 2)  # [B, C, T]
        features = self.temporal_conv(features)
        features = features.transpose(1, 2)  # [B, T, C]
        
        return features


class MotionDecoder(nn.Module):
    """运动解码器"""
    
    def __init__(self, hidden_dim=512, num_joints=22):
        super().__init__()
        self.num_joints = num_joints
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joints * 3)
        )
    
    def forward(self, features):
        return self.decoder(features)


class MotionGPT(nn.Module):
    """
    MotionGPT: 人体运动生成模型
    
    将人体运动建模为语言生成任务
    """
    
    def __init__(
        self,
        num_joints=22,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        vocab_size=50000
    ):
        super().__init__()
        
        # 运动编码器
        self.motion_encoder = MotionEncoder(num_joints, hidden_dim)
        
        # 文本嵌入
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 运动解码器
        self.motion_decoder = MotionDecoder(hidden_dim, num_joints)
        
        # 文本解码器 (用于语言建模)
        self.text_decoder = nn.Linear(hidden_dim, vocab_size)
    
    def generate_motion(self, text_tokens, max_length=120):
        """
        文本生成运动
        
        Args:
            text_tokens: [B, L] 文本 token
            max_length: 运动序列长度
        
        Returns:
            motion: [B, T, J, 3]
        """
        B = text_tokens.shape[0]
        
        # 文本嵌入
        text_embeds = self.text_embedding(text_tokens)
        
        # 位置编码
        text_embeds = text_embeds + self.pos_embedding[:, :text_embeds.shape[1], :]
        
        # 初始化运动 token
        motion_tokens = torch.zeros(B, 1, text_embeds.shape[-1], device=text_tokens.device)
        
        generated_motion = []
        
        for t in range(max_length):
            # 拼接文本和运动 tokens
            combined = torch.cat([text_embeds, motion_tokens], dim=1)
            
            # Transformer
            output = self.transformer(combined)
            
            # 取最后一个位置预测
            next_token = output[:, -1:]
            
            # 预测运动
            motion = self.motion_decoder(next_token)
            generated_motion.append(motion)
            
            # 更新运动 tokens
            motion_tokens = torch.cat([motion_tokens, next_token], dim=1)
        
        # 重组运动序列
        motion = torch.stack(generated_motion, dim=1)
        motion = motion.view(motion.shape[0], motion.shape[1], self.num_joints, 3)
        
        return motion
    
    def forward(self, motion, text_tokens=None):
        """
        前向传播
        
        Args:
            motion: [B, T, J, 3] 运动序列
            text_tokens: [B, L] 文本 token (可选)
        """
        # 编码运动
        motion_features = self.motion_encoder(motion)
        
        if text_tokens is not None:
            # 文本嵌入
            text_features = self.text_embedding(text_tokens)
            
            # 融合
            combined = torch.cat([text_features, motion_features], dim=1)
            
            # Transformer
            output = self.transformer(combined)
            
            # 文本预测 (语言建模)
            text_logits = self.text_decoder(output[:, :text_features.shape[1]])
        else:
            output = self.transformer(motion_features)
            text_logits = None
        
        # 运动预测
        motion_logits = self.motion_decoder(output[:, text_tokens.shape[1] if text_tokens else 0:])
        
        return motion_logits, text_logits


def create_motion_gpt():
    return MotionGPT(num_joints=22, hidden_dim=512, num_layers=8)
