"""
Autonomous Driving 实现

本实现为第三方复现，仅供参考
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVEncoder(nn.Module):
    """Bird's Eye View 编码器"""
    
    def __init__(self, in_channels=3, bev_dim=256):
        super().__init__()
        
        # 多尺度特征提取
        self.backbone = nn.Sequential(
            # Hight: 192, Stride: 4
            nn.Conv2d(in_channels, 32, 7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, bev_dim, 3, stride=2, padding=1)
        )
        
        # Transformer 增强
        self.bev_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=bev_dim, nhead=8, batch_first=True),
            num_layers=4
        )
    
    def forward(self, images):
        """
        Args:
            images: [B, C, H, W] 多视角图像
        
        Returns:
            bev: [B, bev_dim, H_bev, W_bev] BEV 特征
        """
        # 特征提取
        features = self.backbone(images)
        
        # 转换为序列
        B, C, H, W = features.shape
        features_flat = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Transformer
        features_flat = self.bev_transformer(features_flat)
        
        # 恢复空间结构
        bev = features_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return bev


class TransformerDecoder(nn.Module):
    """时序 Transformer 解码器"""
    
    def __init__(self, bev_dim=256, hidden_dim=512, num_layers=6):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=bev_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # 运动查询
        self.motion_queries = nn.Parameter(torch.randn(1, 100, bev_dim))
    
    def forward(self, bev_features, goal):
        """
        Args:
            bev_features: [B, C, H, W] 当前 BEV
            goal: [B, 3] 目标位置 (x, y, yaw)
        """
        B = bev_features.shape[0]
        
        # Flatten BEV
        bev_flat = bev_features.flatten(2).transpose(1, 2)
        
        # 扩展 motion queries
        queries = self.motion_queries.expand(B, -1, -1)
        
        # 解码
        for layer in self.layers:
            queries = layer(queries, bev_flat)
        
        return queries


class TrajectoryPlanner(nn.Module):
    """轨迹规划器"""
    
    def __init__(self, bev_dim=256, num_points=50):
        super().__init__()
        
        self.num_points = num_points
        
        self.planner = nn.Sequential(
            nn.Linear(bev_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_points * 2)  # x, y for each point
        )
        
        # 代价函数
        self.cost_head = nn.Linear(bev_dim, 1)
    
    def forward(self, queries, goal):
        """
        Args:
            queries: [B, N, C] 运动查询
            goal: [B, 3] 目标位置
        
        Returns:
            trajectory: [B, num_points, 2] 规划轨迹
        """
        # 聚合查询特征
        query_features = queries.mean(dim=1)
        
        # 加入目标信息
        query_features = query_features + goal
        
        # 预测轨迹
        trajectory = self.planner(query_features)
        trajectory = trajectory.view(-1, self.num_points, 2)
        
        return trajectory


class PIDController(nn.Module):
    """PID 控制器"""
    
    def __init__(self, kp=1.0, ki=0.1, kd=0.05):
        super().__init__()
        self.kp = kp
        self.ki = ki
        self.kd = kd
    
    def forward(self, trajectory, current_state):
        """
        Args:
            trajectory: [B, N, 2] 规划轨迹
            current_state: [B, 3] 当前状态 (x, y, yaw)
        
        Returns:
            control: [B, 2] 控制量 (steering, throttle)
        """
        # 简化的 PID 控制
        target = trajectory[:, 0]  # 第一个目标点
        
        #  steering = kp * error
        error = target - current_state[:, :2]
        steering = self.kp * error[:, 0]
        
        # throttle = constant
        throttle = torch.ones_like(steering) * 0.5
        
        return torch.stack([steering, throttle], dim=-1)


class End2EndDriving(nn.Module):
    """
    端到端自动驾驶模型
    
    从视觉输入直接输出控制信号
    """
    
    def __init__(
        self,
        num_cameras=8,
        bev_dim=256,
        hidden_dim=512
    ):
        super().__init__()
        
        # 多相机处理
        self.camera_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, 7, stride=2, padding=3),
                nn.ReLU()
            )
            for _ in range(num_cameras)
        ])
        
        # BEV 编码器
        self.bev_encoder = BEVEncoder(in_channels=32 * num_cameras, bev_dim=bev_dim)
        
        # 时序建模
        self.temporal_model = TransformerDecoder(bev_dim=bev_dim, hidden_dim=hidden_dim)
        
        # 轨迹规划
        self.planner = TrajectoryPlanner(bev_dim=bev_dim)
        
        # 控制器
        self.controller = PIDController()
    
    def forward(self, images, goal=None):
        """
        端到端前向传播
        
        Args:
            images: [B, num_cameras, 3, H, W] 多视角图像
            goal: [B, 3] 目标位置 (可选)
        
        Returns:
            control: [B, 2] 控制信号 (steering, throttle)
        """
        B, N, C, H, W = images.shape
        
        # 编码每个相机
        camera_features = []
        for i in range(N):
            feat = self.camera_encoder[i](images[:, i])
            camera_features.append(feat)
        
        # 拼接所有相机特征
        combined = torch.cat(camera_features, dim=1)
        
        # BEV 编码
        bev_features = self.bev_encoder(combined)
        
        # 时序建模
        queries = self.temporal_model(bev_features, goal if goal is not None else torch.zeros(B, 3, device=images.device))
        
        # 轨迹规划
        trajectory = self.planner(queries, goal if goal is not None else torch.zeros(B, 3, device=images.device))
        
        # 控制器
        control = self.controller(trajectory, torch.zeros(B, 3, device=images.device))
        
        return control, trajectory
    
    def drive(self, images, goal):
        """
        推理模式
        
        Args:
            images: 当前视觉输入
            goal: 目标位置
        
        Returns:
            control: 控制信号
        """
        self.eval()
        with torch.no_grad():
            control, _ = self.forward(images.unsqueeze(0), goal.unsqueeze(0))
        return control[0]


def create_autonomous_driving_model():
    """创建自动驾驶模型"""
    return End2EndDriving(num_cameras=8, bev_dim=256)
