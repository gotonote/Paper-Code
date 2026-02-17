"""
分子动力学模拟工具
用于原子级别扩散模型的训练和推理
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
import itertools


class Atom:
    """原子类"""
    def __init__(self, element: str, position: np.ndarray, velocity: np.ndarray = None):
        self.element = element
        self.position = position.astype(np.float32)
        self.velocity = velocity if velocity is not None else np.zeros(3)
        self.force = np.zeros(3)
        self.mass = self._get_mass(element)
        
    def _get_mass(self, element: str) -> float:
        masses = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.065}
        return masses.get(element, 1.0)
    
    def update(self, dt: float, force: np.ndarray):
        """更新位置和速度"""
        self.force = force
        acc = force / self.mass
        self.velocity += acc * dt
        self.position += self.velocity * dt


class Molecule:
    """分子类"""
    def __init__(self, atoms: List[Atom], bonds: List[Tuple[int, int]] = None):
        self.atoms = atoms
        self.bonds = bonds if bonds else []
        
    def get_positions(self) -> np.ndarray:
        return np.array([a.position for a in self.atoms])
    
    def get_forces(self) -> np.ndarray:
        return np.array([a.force for a in self.atoms])
    
    def compute_energy(self) -> float:
        """计算分子能量"""
        # 简化的能量计算
        energy = 0.0
        
        # 键能
        for i, j in self.bonds:
            r = np.linalg.norm(self.atoms[i].position - self.atoms[j].position)
            energy += 0.5 * k_bond * (r - r0) ** 2
            
        # 范德华力
        for i, j in itertools.combinations(range(len(self.atoms)), 2):
            r = np.linalg.norm(self.atoms[i].position - self.atoms[j].position)
            if r > 0:
                energy += 4 * epsilon * (sigma**12 / r**12 - sigma**6 / r**6)
                
        return energy


class ForceField:
    """力场"""
    def __init__(self):
        self.k_bond = 300.0  # 键力常数
        self.r0 = 1.5  # 平衡键长
        self.epsilon = 0.1  # 势阱深度
        self.sigma = 3.4  # 范德华半径
        
    def compute_forces(self, molecule: Molecule) -> np.ndarray:
        """计算力"""
        forces = np.zeros((len(molecule.atoms), 3))
        
        # 键力
        for i, j in molecule.bonds:
            r_vec = molecule.atoms[j].position - molecule.atoms[i].position
            r = np.linalg.norm(r_vec)
            
            if r > 0:
                force_mag = self.k_bond * (r - self.r0)
                force_dir = r_vec / r
                forces[i] += force_mag * force_dir
                forces[j] -= force_mag * force_dir
                
        # 非键力
        for i, j in itertools.combinations(range(len(molecule.atoms)), 2):
            r_vec = molecule.atoms[j].position - molecule.atoms[i].position
            r = np.linalg.norm(r_vec)
            
            if r > 0.1:  # 避免数值问题
                r_norm = r / self.sigma
                f_mag = 24 * self.epsilon * (2 * r_norm**-13 - r_norm**-7) / r
                force_dir = r_vec / r
                
                forces[i] -= f_mag * force_dir
                forces[j] += f_mag * force_dir
                
        return forces


class MolecularDynamics:
    """分子动力学模拟器"""
    def __init__(self, dt: float = 0.001, force_field: ForceField = None):
        self.dt = dt
        self.force_field = force_field if force_field else ForceField()
        
    def step(self, molecule: Molecule):
        """模拟一步"""
        # 计算力
        forces = self.force_field.compute_forces(molecule)
        
        # 更新原子
        for i, atom in enumerate(molecule.atoms):
            atom.update(self.dt, forces[i])
            
    def simulate(self, molecule: Molecule, num_steps: int) -> List[np.ndarray]:
        """运行模拟"""
        trajectories = [molecule.get_positions()]
        
        for _ in range(num_steps):
            self.step(molecule)
            trajectories.append(molecule.get_positions())
            
        return trajectories


# 常用参数
k_bond = 300.0
r0 = 1.5
epsilon = 0.1
sigma = 3.4


def create_water_molecule() -> Molecule:
    """创建水分子"""
    atoms = [
        Atom('O', np.array([0.0, 0.0, 0.0])),
        Atom('H', np.array([0.96, 0.0, 0.0])),
        Atom('H', np.array([-0.24, 0.93, 0.0]))
    ]
    bonds = [(0, 1), (0, 2)]
    return Molecule(atoms, bonds)


def run_simulation(num_steps: int = 1000):
    """运行示例模拟"""
    mol = create_water_molecule()
    md = MolecularDynamics(dt=0.001)
    
    trajs = md.simulate(mol, num_steps)
    
    return trajs
