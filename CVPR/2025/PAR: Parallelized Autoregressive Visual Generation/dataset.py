"""
视觉生成数据集
用于并行自回归视觉生成
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os


class VisualGenerationDataset(Dataset):
    """视觉生成数据集"""
    def __init__(self, image_dir: str, image_size: int = 256, tokenizers: dict = None):
        self.image_dir = image_dir
        self.image_size = image_size
        self.tokenizers = tokenizers
        
        self.image_files = []
        if os.path.exists(image_dir):
            self.image_files = [f for f in os.listdir(image_dir) 
                              if f.endswith(('.jpg', '.png', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        
        # 转换为tensor
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # CHW
        
        return {'image': image_tensor, 'filename': self.image_files[idx]}


class TokenCollator:
    """Token整理器"""
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch):
        images = torch.stack([item['image'] for item in batch])
        
        return {
            'images': images,
            'batch_size': images.size(0)
        }


def create_dataloader(image_dir: str, batch_size: int = 4, 
                     image_size: int = 256):
    """创建视觉生成数据加载器"""
    dataset = VisualGenerationDataset(image_dir, image_size)
    collator = TokenCollator()
    
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collator
    )
