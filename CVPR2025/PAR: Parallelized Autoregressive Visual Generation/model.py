"""
PAR: Parallelized Autoregressive Visual Generation
并行自回归视觉生成
"""

import torch
import torch.nn as nn

class PARModel(nn.Module):
    def __init__(self, vocab_size=8192, embed_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8),
            num_layers=12
        )
        
        self.head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, tokens):
        emb = self.embedding(tokens)
        out = self.decoder(emb)
        return self.head(out)
    
    def generate_parallel(self, batch_size, max_len):
        generated = torch.zeros(batch_size, max_len, dtype=torch.long, device=next(self.parameters()).device)
        for i in range(max_len):
            logits = self.forward(generated[:, :i+1])
            next_token = logits[:, -1].argmax(dim=-1)
            generated[:, i] = next_token
        return generated
