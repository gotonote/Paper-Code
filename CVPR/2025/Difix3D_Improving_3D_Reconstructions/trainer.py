"""
Difix3D 训练代码
包含完整的训练循环和评估
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, List, Optional
import os


class DiffusionTrainer:
    """扩散模型训练器"""
    def __init__(self, model, lr: float = 1e-4, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        self.step = 0
        
    def train_step(self, batch: Dict) -> Dict:
        """单步训练"""
        self.model.train()
        
        images = batch['image'].to(self.device)
        batch_size = images.shape[0]
        
        # 随机时间步
        t = torch.randint(0, 1000, (batch_size,), device=self.device)
        
        # 随机噪声
        noise = torch.randn_like(images)
        
        # 加噪
        alpha_bar = self._get_alpha_bar(t, images.shape)
        noisy_images = alpha_bar.view(-1, 1, 1, 1) * images + (1 - alpha_bar.view(-1, 1, 1, 1)).sqrt() * noise
        
        # 预测噪声
        predicted_noise = self.model(noisy_images, t / 1000.0)
        
        # 损失
        loss = F.mse_loss(predicted_noise, noise)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.step += 1
        
        return {'loss': loss.item()}
    
    def _get_alpha_bar(self, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """获取alpha_bar"""
        return torch.exp(-0.1 * t.float())
    
    @torch.no_grad()
    def sample(self, shape: tuple, num_steps: int = 50) -> torch.Tensor:
        """采样"""
        self.model.eval()
        
        x = torch.randn(shape, device=self.device)
        
        for i in range(num_steps):
            t = torch.full((shape[0],), num_steps - i, device=self.device)
            predicted = self.model(x, t / 1000.0)
            
            alpha = 0.1
            x = x - alpha * predicted / num_steps
            x = x + 0.01 * torch.randn_like(x)
            
        return x
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']


class Evaluator:
    """评估器"""
    def __init__(self, model, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """评估"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            t = torch.randint(0, 1000, (images.shape[0],), device=self.device)
            noise = torch.randn_like(images)
            
            alpha_bar = torch.exp(-0.1 * t.float()).view(-1, 1, 1, 1)
            noisy = alpha_bar * images + (1 - alpha_bar.sqrt()) * noise
            
            predicted = self.model(noisy, t.float() / 1000.0)
            loss = F.mse_loss(predicted, noise)
            
            total_loss += loss.item()
            num_batches += 1
            
        return {'avg_loss': total_loss / num_batches}


def train(config: dict):
    """训练入口"""
    from model import Difix3DModel
    
    model = Difix3DModel()
    trainer = DiffusionTrainer(model, lr=config.get('lr', 1e-4))
    
    print(f"开始训练，设备: {trainer.device}")
    
    for epoch in range(config.get('epochs', 100)):
        metrics = trainer.train_step({'image': torch.randn(4, 3, 256, 256)})
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}")
            
    trainer.save(config.get('save_path', 'model.pt'))
    print("训练完成")
    
    return model
