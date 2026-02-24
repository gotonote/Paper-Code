import torch
import torch.nn as nn

class AIFeedback(nn.Module):
    """AI反馈模块"""
    
    def __init__(self, llm_dim=768):
        super().__init__()
        self.score_head = nn.Linear(llm_dim, 1)
        
    def compute_feedback(self, responses, images):
        """计算AI反馈分数"""
        scores = self.score_head(responses)
        return torch.sigmoid(scores)
