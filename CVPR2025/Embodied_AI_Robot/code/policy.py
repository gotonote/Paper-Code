"""
Embodied AI Robot Policy 实现

本实现为第三方复现，仅供参考
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class VisionEncoder(nn.Module):
    """视觉编码器"""
    
    def __init__(self, vision_dim=768):
        super().__init__()
        
        # 简单的 CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.projection = nn.Linear(128, vision_dim)
    
    def forward(self, images):
        features = self.backbone(images)
        features = features.flatten(1)
        return self.projection(features)


class ActionHead(nn.Module):
    """动作输出头"""
    
    def __init__(self, input_dim=768, action_dim=7, num_horizon=8):
        super().__init__()
        
        self.num_horizon = num_horizon
        self.action_dim = action_dim
        
        # 预测未来 N 步动作
        self.action_pred = nn.Linear(input_dim, num_horizon * action_dim)
    
    def forward(self, features):
        actions = self.action_pred(features)
        actions = actions.view(-1, self.num_horizon, self.action_dim)
        
        # 分离位置和旋转
        position = actions[:, :, :3]  # x, y, z
        rotation = actions[:, :, 3:6]  # roll, pitch, yaw
        gripper = actions[:, :, 6:]  # gripper open/close
        
        return position, rotation, gripper


class RobotPolicy(nn.Module):
    """
    机器人操作策略网络
    
    端到端的语言-视觉-动作映射
    """
    
    def __init__(
        self,
        vision_dim=768,
        language_dim=768,
        hidden_dim=768,
        action_dim=7,
        num_horizon=8
    ):
        super().__init__()
        
        # 视觉编码器
        self.vision_encoder = VisionEncoder(vision_dim)
        
        # 语言编码器
        self.language_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        # 跨模态融合
        self.fusion = nn.MultiheadAttention(
            embed_dim=vision_dim + language_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 动作头
        self.action_head = ActionHead(
            vision_dim + language_dim,
            action_dim,
            num_horizon
        )
    
    def forward(self, images, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            images: [B, C, H, W] 机器人视角图像
            input_ids: [B, L] 语言指令 token ids
            attention_mask: [B, L] attention mask
        
        Returns:
            position: [B, N, 3] 末端执行器位置
            rotation: [B, N, 3] 末端执行器旋转
            gripper: [B, N, 1] 夹爪开合状态
        """
        # 视觉编码
        vision_features = self.vision_encoder(images)  # [B, vision_dim]
        vision_features = vision_features.unsqueeze(1)  # [B, 1, vision_dim]
        
        # 语言编码
        language_outputs = self.language_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        language_features = language_outputs.last_hidden_state  # [B, L, language_dim]
        
        # 融合视觉和语言
        combined = torch.cat([vision_features, language_features], dim=1)
        
        # 自注意力融合
        fused, _ = self.fusion(combined, combined, combined)
        
        # 使用 CLS token 的输出
        cls_output = fused[:, 0, :]  # [B, vision_dim + language_dim]
        
        # 预测动作
        position, rotation, gripper = self.action_head(cls_output)
        
        return position, rotation, gripper
    
    def act(self, image, instruction):
        """
        单步动作推理
        
        Args:
            image: 单张图像
            instruction: 语言指令
        
        Returns:
            action: 动作字典
        """
        self.eval()
        with torch.no_grad():
            # Tokenize
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            inputs = tokenizer(instruction, return_tensors="pt")
            
            # Forward
            position, rotation, gripper = self.forward(
                image.unsqueeze(0),
                inputs.input_ids,
                inputs.attention_mask
            )
            
            # 取第一帧动作
            action = {
                'position': position[0, 0].cpu().numpy(),
                'rotation': rotation[0, 0].cpu().numpy(),
                'gripper': torch.sigmoid(gripper[0, 0]).cpu().numpy()
            }
        
        return action


def create_robot_policy():
    """创建机器人策略网络"""
    return RobotPolicy()
