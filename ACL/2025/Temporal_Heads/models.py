"""Temporal Heads模型"""
import torch
import torch.nn as nn

class TemporalHeadModel(nn.Module):
    def __init__(self, vocab_size=30000, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.temporal_head = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.classifier = nn.Linear(embed_dim, 2)
        
    def forward(self, x):
        emb = self.embedding(x)
        attn_out, _ = self.temporal_head(emb, emb, emb)
        return self.classifier(attn_out.mean(dim=1))
