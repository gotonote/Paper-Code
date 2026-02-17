"""
LADDER训练代码
"""
import torch

def train_ladder():
    model = LADDERModel(input_dim=784)
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(10):
        print(f"Epoch {epoch+1}")
        
    return model
