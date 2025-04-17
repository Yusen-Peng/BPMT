import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import collate_fn_pairs
from tqdm import tqdm
from typing import Tuple
from first_phase_baseline import BaseT1, mask_keypoints

def load_T1(model_path: str, num_joints: int = 14, d_model: int = 128, nhead: int = 4, num_layers: int = 2, freeze: bool = True,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> BaseT1:
    """
        loads a BaseT1 model from a checkpoint
    """

    model = BaseT1(num_joints=num_joints, d_model=d_model, nhead=nhead, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # optionally freeze the model parameters
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # move model to device and return the model
    return model.to(device)

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
        A simple baseline transformer model for reconstructing masked keypoints (second stage).

    """
    def __init__(self, out_dim_A: int, out_dim_B: int, d_model=128, nhead=4, num_layers=2):
        super(BaseT2, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )

        # separate heads for each modality
        self.headA = nn.Linear(d_model, out_dim_A)
        self.headB = nn.Linear(d_model, out_dim_B)

    def forward(self, x, T_A):
        encoded = self.encoder(x)

        # shape (B, T_A, out_dim_A)
        reconsA = self.headA(encoded[:, :T_A, :])
        # shape (B, T_B, out_dim_B)
        reconsB = self.headB(encoded[:, T_A:, :])

        return reconsA, reconsB
    
    def encode(self, x, T_A):
        """
            Encodes the input sequence using the transformer encoder.
            Returns the encoded features.
        """
        encoded = self.encoder(x)
        
        encoded_A = encoded[:, :T_A, :]
        encoded_B = encoded[:, T_A:, :]
        return encoded_A, encoded_B        


def train_T2(
            modality_name_A,
            modality_name_B,
            train_pairwise_dataset,
            val_pairwise_dataset, 
            model_pathA: str, 
            model_pathB: str, 
            num_joints_A: int,
            num_joints_B: int,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 2,
            num_epochs: int = 50,
            batch_size: int = 16,
            lr: float = 1e-4,
            mask_ratio: float = 0.15,
            freeze_T1: bool = True,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        ) -> Tuple[BaseT2, CrossAttention]:
    """
        second-stage pretraining using pretrained T1 models for two modalities.

    """

    # load pretrained T1 encoders
    modality_A = load_T1(model_pathA, num_joints_A, d_model, nhead, num_layers, freeze_T1, device)
    modality_B = load_T1(model_pathB, num_joints_B, d_model, nhead, num_layers, freeze_T1, device)

    # intialize the cross-attention and transformer encoder
    cross_attn = CrossAttention(d_model=d_model, nhead=nhead)
    model_T2 = BaseT2(
        out_dim_A=num_joints_A * 2,
        out_dim_B=num_joints_B * 2,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    cross_attn.to(device)
    model_T2.to(device)

    # Dataloader
    train_loader = DataLoader(
        train_pairwise_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_pairs
    )

    val_loader = DataLoader(
        val_pairwise_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_pairs
    )

    # optimize both the cross-attention and the transformer encoder
    optimizer = optim.Adam(list(cross_attn.parameters()) + list(model_T2.parameters()), lr=lr)
    criterion = nn.MSELoss(reduction='none')

    # we also need to visualize both the train and val loss
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        modality_A.eval()
        modality_B.eval()
        cross_attn.train()
        model_T2.train()

        train_loss = 0.0

        for sequences_A, sequences_B in train_loader:

            sequences_A = sequences_A.float().to(device)
            sequences_B = sequences_B.float().to(device)
        
            # # masking
            # maskedA, maskA = mask_keypoints(sequences_A, mask_ratio)
            # maskedB, maskB = mask_keypoints(sequences_B, mask_ratio)

            # encoding
            # featsA = modality_A.encode(maskedA)
            # featsB = modality_B.encode(maskedB)
            featsA = modality_A.encode(sequences_A)
            featsB = modality_B.encode(sequences_B)

            # cross-attention
            A_attends_B = cross_attn(featsA, featsB, featsB)
            B_attends_A = cross_attn(featsB, featsA, featsA)

            # concatenate + T2 forward pass
            concatenated = torch.cat([A_attends_B, B_attends_A], dim=1)
            reconsA, reconsB = model_T2(concatenated, sequences_A.size(1))        

            # compute the reconstruction loss
            lossA = criterion(reconsA, sequences_A)
            lossB = criterion(reconsB, sequences_B)

            # # again, we only do MSE on masked positions
            # # we also need to broadcast mask to match the shape 
            # maskA = maskA.unsqueeze(-1).expand_as(lossA)
            # maskB = maskB.unsqueeze(-1).expand_as(lossB)
            # lossA = (lossA * maskA).sum() / (maskA.sum() + 1e-8)
            # lossB = (lossB * maskB).sum() / (maskB.sum() + 1e-8)

            lossA = lossA.mean()
            lossB = lossB.mean()

            # average loss
            loss = (lossA + lossB) * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sequences_A.size(0)
        
        # average epoch loss
        train_loss /= len(train_pairwise_dataset)

        # validation step
        cross_attn.eval()
        model_T2.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences_A, sequences_B in val_loader:
                sequences_A = sequences_A.float().to(device)
                sequences_B = sequences_B.float().to(device)

                # masking
                # maskedA, maskA = mask_keypoints(sequences_A, mask_ratio)
                # maskedB, maskB = mask_keypoints(sequences_B, mask_ratio)

                # encoding
                # featsA = modality_A.encode(maskedA)
                # featsB = modality_B.encode(maskedB)
                featsA = modality_A.encode(sequences_A)
                featsB = modality_B.encode(sequences_B)
                
                # cross-attention
                A_attends_B = cross_attn(featsA, featsB, featsB)
                B_attends_A = cross_attn(featsB, featsA, featsA)

                # concatenate + T2 forward pass
                concatenated = torch.cat([A_attends_B, B_attends_A], dim=1)
                reconsA, reconsB = model_T2(concatenated, sequences_A.size(1))

                # compute the reconstruction loss
                lossA = criterion(reconsA, sequences_A)
                lossB = criterion(reconsB, sequences_B)

                # again, we only do MSE on masked positions
                # we also need to broadcast mask to match the shape 
                # maskA = maskA.unsqueeze(-1).expand_as(lossA)
                # maskB = maskB.unsqueeze(-1).expand_as(lossB)
                # lossA = (lossA * maskA).sum() / (maskA.sum() + 1e-8)
                # lossB = (lossB * maskB).sum() / (maskB.sum() + 1e-8)


                lossA = lossA.mean()
                lossB = lossB.mean()

                # average loss
                loss = (lossA + lossB) * 0.5

                val_loss += loss.item() * sequences_A.size(0)
        
        # average epoch loss
        val_loss /= len(val_pairwise_dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{modality_name_A} ~ {modality_name_B} - Train and Val Loss')
    plt.savefig(f'figures/{modality_name_A}_{modality_name_B}_train_val_loss.png')

    return model_T2, cross_attn
