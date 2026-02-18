"""
自动驾驶轨迹规划器
包含路径规划和速度规划
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class TrajectoryPlanner:
    """轨迹规划器"""
    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size
        self.resolution = 0.1  # 10cm per grid
        
    def plan(self, start: np.ndarray, goal: np.ndarray, 
             obstacles: List[np.ndarray]) -> np.ndarray:
        """路径规划"""
        path = self._hybrid_astar(start, goal, obstacles)
        return self._smooth_path(path)
    
    def _hybrid_astar(self, start, goal, obstacles):
        """混合A*算法"""
        path = [start]
        current = start.copy()
        
        for _ in range(100):
            direction = goal - current
            distance = np.linalg.norm(direction)
            
            if distance < 0.5:
                break
                
            step = direction / distance * 0.5
            current = current + step
            path.append(current.copy())
            
        return np.array(path)
    
    def _smooth_path(self, path: np.ndarray) -> np.ndarray:
        """路径平滑"""
        if len(path) < 3:
            return path
            
        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            avg = (path[i-1] + path[i] + path[i+1]) / 3
            smoothed.append(avg)
        smoothed.append(path[-1])
        
        return np.array(smoothed)


class SpeedProfile:
    """速度剖面规划"""
    def __init__(self, max_speed: float = 10.0, max_accel: float = 3.0):
        self.max_speed = max_speed
        self.max_accel = max_accel
        
    def compute(self, trajectory: np.ndarray, 
                curvature: Optional[np.ndarray] = None) -> np.ndarray:
        """计算速度剖面"""
        n = len(trajectory)
        speeds = np.zeros(n)
        
        for i in range(n):
            # 基于曲率的速度限制
            if curvature is not None and i < len(curvature):
                speed_limit = self.max_speed / (1 + abs(curvature[i]) * 2)
            else:
                speed_limit = self.max_speed
                
            # 距离终点的速度限制
            dist_to_end = np.linalg.norm(trajectory[i:] - trajectory[-1]) if i < n-1 else 0
            speed_limit = min(speed_limit, np.sqrt(2 * self.max_accel * dist_to_end))
            
            speeds[i] = speed_limit
            
        return speeds


class MotionPlanner:
    """运动规划器"""
    def __init__(self):
        self.trajectory_planner = TrajectoryPlanner()
        self.speed_profile = SpeedProfile()
        
    def plan(self, start_state: dict, goal_state: dict, 
             obstacles: List[dict]) -> dict:
        """完整运动规划"""
        start_pos = np.array(start_state['position'])
        goal_pos = np.array(goal_state['position'])
        
        obs_pos = [np.array(o['position']) for o in obstacles]
        
        # 路径规划
        trajectory = self.trajectory_planner.plan(start_pos, goal_pos, obs_pos)
        
        # 速度规划
        speeds = self.speed_profile.compute(trajectory)
        
        # 生成时间参数
        times = self._compute_times(trajectory, speeds)
        
        return {
            'trajectory': trajectory,
            'speeds': speeds,
            'times': times
        }
        
    def _compute_times(self, traj: np.ndarray, speeds: np.ndarray) -> np.ndarray:
        """计算时间参数"""
        n = len(traj)
        times = np.zeros(n)
        
        for i in range(1, n):
            distance = np.linalg.norm(traj[i] - traj[i-1])
            avg_speed = (speeds[i] + speeds[i-1]) / 2
            if avg_speed > 0.1:
                times[i] = times[i-1] + distance / avg_speed
            else:
                times[i] = times[i-1] + 0.1
                
        return times


class LateralController:
    """横向控制器 - LQR"""
    def __init__(self, lqr_gain: np.ndarray = None):
        if lqr_gain is None:
            self.K = np.array([1.0, 2.0, 0.5])
        else:
            self.K = lqr_gain
            
    def compute(self, error: dict) -> float:
        """计算横向控制量"""
        state = np.array([error['lateral'], error['heading'], error['curvature']])
        steering = -self.K @ state
        return np.clip(steering, -0.5, 0.5)


class LongitudinalController:
    """纵向控制器 - PID"""
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.05):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0
        
    def compute(self, target_speed: float, current_speed: float, dt: float) -> float:
        """计算纵向控制量"""
        error = target_speed - current_speed
        
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.prev_error = error
        
        return np.clip(control, -1.0, 1.0)


def create_motion_planner():
    return MotionPlanner()
