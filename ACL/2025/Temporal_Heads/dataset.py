"""
时序数据处理
用于时间序列分析
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Optional


class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    def __init__(self, data: np.ndarray, seq_len: int = 100, stride: int = 1):
        self.data = torch.from_numpy(data).float()
        self.seq_len = seq_len
        self.stride = stride
        
    def __len__(self):
        return max(1, (len(self.data) - self.seq_len) // self.stride)
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        return self.data[start:end]


class TemporalAugmentation:
    """时序数据增强"""
    def __init__(self):
        pass
    
    def jitter(self, x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
        return x + torch.randn_like(x) * sigma
    
    def scaling(self, x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        factor = torch.randn(x.size(0)) * sigma + 1.0
        return x * factor.view(-1, 1)
    
    def permutation(self, x: torch.Tensor, max_segments: int = 5) -> torch.Tensor:
        orig_steps = np.arange(x.size(1))
        num_segs = np.random.randint(1, max_segments)
        
        if num_segs > 1:
            split_points = np.random.choice(x.size(1)-2, num_segs-1, replace=False)
            split_points.sort()
            splits = np.split(orig_steps, split_points)
            np.random.shuffle(splits)
            permuted_indices = np.concatenate(splits)
            return x[:, permuted_indices]
        return x


def create_time_series_loader(data, batch_size: int = 32, seq_len: int = 100):
    """创建时序数据加载器"""
    dataset = TimeSeriesDataset(data, seq_len)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
