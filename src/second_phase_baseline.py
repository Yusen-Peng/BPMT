import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import collate_fn_batch_padding
from first_phase_baseline import BaseT1



def load_and_fuse(model_pathA: str, model_pathB: str, xA: torch.Tensor, xB: torch.Tensor,
    num_joints=14,
    d_model=128,
    nhead=4,
    num_layers=2,
    freeze_baseT1: bool = False
) -> torch.Tensor:

    """
    Loads two pretrained BaseT1 models (for two modalities), extracts their encoded features,
    and returns a fused representation.

    Args:
        model_pathA: path to the checkpoint for modality A's BaseT1.
        model_pathB: path to the checkpoint for modality B's BaseT1.
        xA, xB: input tensors for each modality, of shape (B, T, 2*num_joints).
        num_joints, d_model, nhead, num_layers: parameters for BaseT1 model creation.
        freeze_baseT1: if True, sets requires_grad=False on BaseT1 parameters.

    Returns:
        fused: a torch.Tensor containing the fused representation from both modalities.
    """
    # load the baseT1 checkpoints for two modalities
    # load onto CPU first, then move to GPU if available
    modA = BaseT1(num_joints=num_joints, d_model=d_model, nhead=nhead, num_layers=num_layers)
    modB = BaseT1(num_joints=num_joints, d_model=d_model, nhead=nhead, num_layers=num_layers)
    modA.load_state_dict(torch.load(model_pathA, map_location='cpu'))
    modB.load_state_dict(torch.load(model_pathB, map_location='cpu'))

    # we can optionally freeze parameters for the baseT1 models
    if freeze_baseT1:
        for param in modA.parameters():
            param.requires_grad = False
        for param in modB.parameters():
            param.requires_grad = False



    # 4) Extract encoded features
    featsA = modA.get_encoded_features(xA)  # shape (B, T, d_model)
    featsB = modB.get_encoded_features(xB)  # shape (B, T, d_model)

    # 5) Simple Fusion by concatenation along the feature dimension
    # If you'd rather combine along the time dimension, or do some other approach, adjust here.
    # (B, T, d_model) => (B, T, 2*d_model)
    fused = torch.cat([featsA, featsB], dim=2)

    return fused



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


