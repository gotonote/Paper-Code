"""Alpha-SQL训练"""
import torch

class AlphaSQLTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        questions, schemas, sqls = batch
        output = self.model(questions, schemas)
        loss = torch.nn.functional.cross_entropy(output, sqls)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
