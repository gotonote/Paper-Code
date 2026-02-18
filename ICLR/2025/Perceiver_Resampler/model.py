import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceiverResampler(nn.Module):
    '''
    Perceiver Resampler
    '''
    
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 編碼器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 解碼器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        return out


def train_step(model, batch):
    '''訓練步驟'''
    inputs, targets = batch
    outputs = model(inputs)
    loss = F.mse_loss(outputs, targets)
    return loss


if __name__ == "__main__":
    model = PerceiverResampler()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 測試
    x = torch.randn(2, 512)
    out = model(x)
    print(f"Output shape: {out.shape}")
