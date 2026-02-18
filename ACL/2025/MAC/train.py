"""
MAC: Multi-hop Attention for Context-aware Learning
训练代码
"""

import torch
import torch.nn as nn
from model import MACModel

def train_mac():
    """MAC模型训练"""
    model = MACModel(vocab_size=30000, embed_dim=256, num_layers=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(10):
        print(f"Epoch {epoch+1}")
        
    print("训练完成")

if __name__ == "__main__":
    train_mac()
