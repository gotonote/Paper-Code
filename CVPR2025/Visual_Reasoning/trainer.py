"""视觉推理训练代码"""
import torch

class VisualReasoningTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        images, questions, answers = batch
        output = self.model(images, questions)
        loss = torch.nn.functional.cross_entropy(output, answers)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
