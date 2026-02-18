"""具身AI机器人训练代码"""
import torch

class EmbodiedTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        obs, actions = batch
        pred_actions = self.model(obs)
        loss = torch.nn.functional.mse_loss(pred_actions, actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
