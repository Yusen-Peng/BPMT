import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import collate_fn_batch_padding



class CrossAttention(nn.Module):
    """
    Simple cross-attention block that applies multi-head attention between two feature sequences.
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        # cross-attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # layer normalization
        self.norm = nn.LayerNorm(d_model)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:        
        # cross attention
        attn_out, _ = self.cross_attention(Q, K, V)

        # apply layer normalization and dropout, and residual connection
        out = self.norm(Q + self.dropout(attn_out))
        
        return out
