import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import collate_fn_batch_padding
from first_phase_baseline import BaseT1

def load_and_encode(model_pathA: str,
                    model_pathB: str, 
                    xA: torch.Tensor, 
                    xB: torch.Tensor,
                    num_joints=14,
                    d_model=128,
                    nhead=4,
                    num_layers=2,
                    freeze_T1: bool = True,
                    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                ) -> torch.Tensor:

    """
        Loads two pretrained BaseT1 models (for two modalities) and extracts their encoded features.

    """
    # load the baseT1 checkpoints for two modalities
    # load onto CPU first, then move to GPU if available
    modA = BaseT1(num_joints=num_joints, d_model=d_model, nhead=nhead, num_layers=num_layers)
    modB = BaseT1(num_joints=num_joints, d_model=d_model, nhead=nhead, num_layers=num_layers)
    modA.load_state_dict(torch.load(model_pathA, map_location='cpu'))
    modB.load_state_dict(torch.load(model_pathB, map_location='cpu'))

    # we can optionally freeze parameters for the T1 models
    if freeze_T1:
        for param in modA.parameters():
            param.requires_grad = False
        for param in modB.parameters():
            param.requires_grad = False

    # move models and data to the appropriate device
    modA.to(device)
    modB.to(device)
    xA = xA.to(device)
    xB = xB.to(device)
    
    # feature extraction
    feature_A = modA.encode(xA)
    feature_B = modB.encode(xB)
    return feature_A, feature_B


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


class BaseT2(nn.Module):
    """
        A second-stage transformer that processes cross-attended features.
    """
    def __init__(self, num_joints: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super(BaseT2, self).__init__()
        
        # a standard Transformer Encoder stack
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # reconstruction head
        self.reconstruction_head = nn.Linear(d_model, num_joints * 2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.transformer_encoder(x)
        recons = self.reconstruction_head(encoded)
        return recons
    


