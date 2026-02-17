
"""
多视角城市数据集处理
用于多视角城市场景理解
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Optional
import os


class UrbanScene:
    """城市场景"""
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.images = []
        self.poses = []
        self.intrinsics = []
        
    def add_view(self, image: np.ndarray, pose: np.ndarray, K: np.ndarray):
        self.images.append(image)
        self.poses.append(pose)
        self.intrinsics.append(K)


class MultiViewDataset(Dataset):
    """多视角数据集"""
    def __init__(self, data_root: str, split: str = 'train'):
        self.data_root = data_root
        self.split = split
        self.scenes = self._load_scenes()
        
    def _load_scenes(self) -> List[UrbanScene]:
        """加载场景"""
        scenes = []
        scene_dir = os.path.join(self.data_root, self.split)
        
        if os.path.exists(scene_dir):
            for scene_id in os.listdir(scene_dir):
                scene = UrbanScene(scene_id)
                scenes.append(scene)
                
        return scenes
    
    def __len__(self) -> int:
        return len(self.scenes)
    
    def __getitem__(self, idx: int) -> Dict:
        scene = self.scenes[idx]
        
        # 随机选择一个视角作为查询
        query_idx = np.random.randint(0, len(scene.images))
        
        return {
            'query_image': scene.images[query_idx],
            'query_pose': scene.poses[query_idx],
            'query_K': scene.intrinsics[query_idx],
            'scene_id': scene.scene_id
        }


class ViewSelector:
    """视角选择器"""
    def __init__(self):
        self.num_neighbors = 5
        
    def select(self, scene: UrbanScene, query_idx: int) -> List[int]:
        """选择最相关的视角"""
        query_pose = scene.poses[query_idx]
        
        distances = []
        for i, pose in enumerate(scene.poses):
            if i != query_idx:
                dist = np.linalg.norm(pose[:3, 3] - query_pose[:3, 3])
                distances.append((i, dist))
                
        distances.sort(key=lambda x: x[1])
        
        return [d[0] for d in distances[:self.num_neighbors]]


def create_dataloader(data_root: str, batch_size: int = 4, split: str = 'train'):
    """创建数据加载器"""
    dataset = MultiViewDataset(data_root, split)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
