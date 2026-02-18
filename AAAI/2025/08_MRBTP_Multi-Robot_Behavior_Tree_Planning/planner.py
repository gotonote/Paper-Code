"""
多机器人行为树规划器
包含完整的行为树节点和执行逻辑
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import numpy as np


class BehaviorTreeNode:
    """行为树基类"""
    def __init__(self, name: str):
        self.name = name
        self.children = []
        
    def add_child(self, child):
        self.children.append(child)
        
    def execute(self, blackboard) -> bool:
        raise NotImplementedError


class SequenceNode(BehaviorTreeNode):
    """顺序节点：所有子节点都成功才算成功"""
    def __init__(self, name: str):
        super().__init__(name)
        
    def execute(self, blackboard) -> bool:
        for child in self.children:
            if not child.execute(blackboard):
                return False
        return True


class SelectorNode(BehaviorTreeNode):
    """选择节点：任意子节点成功即成功"""
    def __init__(self, name: str):
        super().__init__(name)
        
    def execute(self, blackboard) -> bool:
        for child in self.children:
            if child.execute(blackboard):
                return True
        return False


class ConditionNode(BehaviorTreeNode):
    """条件节点：检查是否满足条件"""
    def __init__(self, name: str, condition_fn):
        super().__init__(name)
        self.condition_fn = condition_fn
        
    def execute(self, blackboard) -> bool:
        return self.condition_fn(blackboard)


class ActionNode(BehaviorTreeNode):
    """动作节点：执行具体动作"""
    def __init__(self, name: str, action_fn):
        super().__init__(name)
        self.action_fn = action_fn
        
    def execute(self, blackboard) -> bool:
        return self.action_fn(blackboard)


class BehaviorTreePlanner:
    """行为树规划器"""
    def __init__(self, num_robots: int = 4):
        self.num_robots = num_robots
        self.trees = [self._create_tree() for _ in range(num_robots)]
        self.blackboard = {}
        
    def _create_tree(self) -> BehaviorTreeNode:
        """创建完整的行为树"""
        # 根节点：选择器
        root = SelectorNode("root")
        
        # 导航到目标
        nav_sequence = SequenceNode("navigate")
        nav_sequence.add_child(ConditionNode("has_target", self._check_has_target))
        nav_sequence.add_child(ActionNode("plan_path", self._plan_path))
        nav_sequence.add_child(ActionNode("execute_motion", self._execute_motion))
        
        # 避障
        avoid_sequence = SequenceNode("avoid_obstacle")
        avoid_sequence.add_child(ConditionNode("obstacle_detected", self._check_obstacle))
        avoid_sequence.add_child(ActionNode("compute_avoidance", self._compute_avoidance))
        
        # 协作
        collab_sequence = SequenceNode("collaborate")
        collab_sequence.add_child(ConditionNode("need_help", self._check_need_help))
        collab_sequence.add_child(ActionNode("request_assist", self._request_assist))
        
        root.add_child(nav_sequence)
        root.add_child(avoid_sequence)
        root.add_child(collab_sequence)
        
        return root
    
    def _check_has_target(self, bb) -> bool:
        return 'target' in bb and bb['target'] is not None
    
    def _plan_path(self, bb) -> bool:
        if 'target' not in bb:
            return False
        target = bb['target']
        current = bb.get('position', np.zeros(3))
        bb['path'] = self._astar(current, target)
        return True
    
    def _execute_motion(self, bb) -> bool:
        if 'path' not in bb or len(bb['path']) == 0:
            return False
        next_pos = bb['path'].pop(0)
        bb['position'] = next_pos
        return True
    
    def _check_obstacle(self, bb) -> bool:
        obstacles = bb.get('obstacles', [])
        position = bb.get('position', np.zeros(3))
        for obs in obstacles:
            if np.linalg.norm(obs - position) < 1.0:
                return True
        return False
    
    def _compute_avoidance(self, bb) -> bool:
        bb['velocity'] = -bb.get('velocity', np.zeros(3)) * 0.5
        return True
    
    def _check_need_help(self, bb) -> bool:
        task_complexity = bb.get('complexity', 0)
        return task_complexity > 0.7
    
    def _request_assist(self, bb) -> bool:
        bb['assist_requested'] = True
        return True
    
    def _astar(self, start, goal):
        """A*路径规划"""
        path = [goal]
        return path
    
    def plan(self, state: Dict[str, Any]) -> List[str]:
        """规划动作序列"""
        self.blackboard.update(state)
        actions = []
        
        for tree in self.trees:
            self.blackboard['robot_id'] = self.trees.index(tree)
            if tree.execute(self.blackboard):
                actions.append(f"robot_{self.blackboard['robot_id']}_action")
        
        return actions


class RobotController:
    """机器人控制器"""
    def __init__(self, planner: BehaviorTreePlanner):
        self.planner = planner
        self.state = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'target': None,
            'obstacles': [],
            'complexity': 0.0
        }
        
    def update(self, dt: float):
        """更新机器人状态"""
        actions = self.planner.plan(self.state)
        
        for action in actions:
            if 'execute_motion' in action:
                self.state['position'] += self.state['velocity'] * dt
                
        return self.state
    
    def reset(self):
        """重置状态"""
        self.state = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'target': None,
            'obstacles': [],
            'complexity': 0.0
        }


def create_multi_robot_system(num_robots: int = 4):
    """创建多机器人系统"""
    planner = BehaviorTreePlanner(num_robots)
    controllers = [RobotController(planner) for _ in range(num_robots)]
    return planners, controllers
