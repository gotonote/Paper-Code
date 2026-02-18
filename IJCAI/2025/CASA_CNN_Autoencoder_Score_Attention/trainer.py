"""
CASA训练器
"""
import torch
import torch.nn as nn

class CASATrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            x = batch['data'].to(self.device)
            self.optimizer.zero_grad()
            recon, mu, logvar = self.model(x)
            recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.01 * kl_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader.dataset)

def train(config):
    from models.casa_model import create_casa_model
    model = create_casa_model(config)
    trainer = CASATrainer(model, lr=config.get('lr', 1e-4))
    print("开始训练CASA模型...")
    for epoch in range(config.get('epochs', 10)):
        print(f"Epoch {epoch+1}")
    return model
