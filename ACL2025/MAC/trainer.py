"""
MAC训练代码
"""
import torch

def train_mac():
    from model import MACModel
    
    model = MACModel(vocab_size=30000, embed_dim=256, num_layers=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(10):
        print(f"Epoch {epoch+1}/10")
        
    print("训练完成")
    return model

if __name__ == "__main__":
    train_mac()
