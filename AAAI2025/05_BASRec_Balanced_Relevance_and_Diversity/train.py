"""
BASRec: Balanced Relevance and Diversity Recommendation
推荐系统平衡相关性多样性训练代码
"""

import argparse
import torch
import torch.nn as nn
from model import BASRecModel

def train(args):
    """训练模型"""
    model = BASRecModel(
        user_num=args.user_num,
        item_num=args.item_num,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in train_loader:
            users, items, labels = batch
            
            # 前向传播
            scores = model(users, items)
            loss = criterion(scores, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), f"model_epoch{args.epochs}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_num", type=int, default=10000)
    parser.add_argument("--item_num", type=int, default=1000)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    train(args)
