import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from penn_utils import collate_fn_batch_padding
from tqdm import tqdm
from typing import Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR

POSITIONAL_UPPER_BOUND = 1000

def collate_fn_stack(batch):
    sequences, labels = zip(*batch)
    return torch.stack([torch.tensor(x) for x in sequences]), torch.tensor(labels)


class BaseT1(nn.Module):
    """
        A simple baseline transformer model for reconstructing masked keypoints.
        The model consists of:
        - Keypoint embedding layer
        - Positional embedding layer
        - Transformer encoder
        - Reconstruction head
        The model is designed to take in sequences of 2D keypoints and reconstruct the masked frames.
    """
    
    def __init__(self, num_joints: int, three_d: bool, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super(BaseT1, self).__init__()
        self.num_joints = num_joints
        self.d_model = d_model
        self.three_d = three_d

        # keypoint embedding
        if three_d:
            self.embedding = nn.Linear(num_joints * 3, d_model)
        else:
            self.embedding = nn.Linear(num_joints * 2, d_model)

        # positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, POSITIONAL_UPPER_BOUND, d_model))

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # reconstruction head (only used during training)
        if three_d:
            self.reconstruction_head = nn.Linear(d_model, num_joints * 3)
        else:   
            self.reconstruction_head = nn.Linear(d_model, num_joints * 2)
        
        
        # MLP reconstruction head (optional)
        # self.reconstruction_head = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, num_joints * 2)
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass of the model with the decoder.
            Args:
                x (torch.Tensor): Input tensor of shape (B, T, num_joints * 2).
            Returns:
                torch.Tensor: Reconstructed tensor of shape (B, T, num_joints * 2).
        """
        B, T, _ = x.shape
        keypoint_embedding = self.embedding(x)
        keypoint_embedding_with_pos = keypoint_embedding + self.pos_embedding[:, :T, :]

        # NOTE: PyTorch Transformer wants shape (T, B, d_model) instead of (B, T, d_model)
        keypoint_embedding_with_pos = keypoint_embedding_with_pos.transpose(0,1)
        encoded = self.transformer_encoder(keypoint_embedding_with_pos)
        encoded = encoded.transpose(0,1)

        recons = self.reconstruction_head(encoded)
        return recons
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
            just encode the input sequence without reconstruction.
        """
        B, T, _ = x.shape
        keypoint_embedding = self.embedding(x)
        keypoint_embedding_with_pos = keypoint_embedding + self.pos_embedding[:, :T, :]

        # NOTE: PyTorch Transformer wants shape (T, B, d_model) instead of (B, T, d_model)
        keypoint_embedding_with_pos = keypoint_embedding_with_pos.transpose(0,1)
        encoded = self.transformer_encoder(keypoint_embedding_with_pos)
        encoded = encoded.transpose(0,1)

        return encoded
    

PAD_IDX = 0.0

def mask_random_frames(inputs: torch.Tensor, mask_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, _ = inputs.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=inputs.device)

    for i in range(B):
        num_to_mask = max(1, int(mask_ratio * T))
        mask_indices = torch.randperm(T, device=inputs.device)[:num_to_mask]
        mask[i, mask_indices] = 1

    masked_inputs = inputs.clone()
    masked_inputs[mask.unsqueeze(-1).expand_as(inputs)] = PAD_IDX

    return masked_inputs, mask


def mask_random_global_joints(inputs: torch.Tensor, num_joints: int, joint_dim: int, mask_ratio: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly masks out a percentage of joint slots globally across all frames.

    Args:
        inputs: Tensor of shape [B, T, C] where C = num_joints * joint_dim
        num_joints: Number of joints per frame
        joint_dim: Number of values per joint (e.g. 2 for (x,y), 3 for (x,y,conf))
        mask_ratio: Percentage of joint slots (T * num_joints) to mask out

    Returns:
        masked_inputs: Same shape as inputs, with PAD_IDX in masked positions
        mask: Boolean tensor of shape [B, T, num_joints], True at masked joints
    """
    B, T, C = inputs.shape
    assert C == num_joints * joint_dim

    masked_inputs = inputs.clone()
    mask = torch.zeros(B, T, num_joints, dtype=torch.bool, device=inputs.device)

    # the total number of slots per sample in the batch
    total_slots = T * num_joints
    num_to_mask = max(1, int(mask_ratio * total_slots))

    for i in range(B):
        indices = torch.randperm(total_slots, device=inputs.device)[:num_to_mask]

        # 1D to 2D index mapping
        t_indices = indices // num_joints
        j_indices = indices % num_joints
        mask[i, t_indices, j_indices] = 1

    # expand mask to [B, T, C] by repeating each joint's mask across joint_dim
    expanded_mask = mask.repeat_interleave(joint_dim, dim=2)  # shape [B, T, C]

    masked_inputs[expanded_mask] = PAD_IDX
    return masked_inputs, mask

def train_T1(masking_strategy, train_dataset, val_dataset, model: BaseT1, num_epochs=50, batch_size=16, lr=1e-4, mask_ratio=0.15, device='cuda'):
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_stack,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_stack,
    )

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    model.to(device)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        for sequences, _ in train_loader:
            sequences = sequences.float().to(device)

              # Masked pretraining
            if mask_ratio is not None:
                if masking_strategy == "frame":
                    masked_inputs, mask = mask_random_frames(sequences, mask_ratio=mask_ratio)
                elif masking_strategy == "global_joint":
                    joint_dim = 3 if model.three_d == True else 2
                    masked_inputs, mask = mask_random_global_joints(sequences, num_joints=model.num_joints, joint_dim=joint_dim, mask_ratio=mask_ratio)                
                else:
                    raise ValueError(f"Unknown masking strategy: {masking_strategy}")
            else:
                masked_inputs = sequences
                mask = torch.ones_like(sequences[..., 0])

            recons = model(masked_inputs)

            loss_matrix = criterion(recons, sequences)

            # loss_matrix already exists
            
            joint_dim = 3 if model.three_d else 2
            if masking_strategy == "frame":
                mask_broadcasted = mask.unsqueeze(-1).expand_as(recons)

            elif masking_strategy == "global_joint":
                mask_broadcasted = mask.repeat_interleave(joint_dim, dim=-1)
            
            masked_loss = loss_matrix * mask_broadcasted
            num_masked = mask_broadcasted.sum()
            loss = masked_loss.sum() / (num_masked + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sequences.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, _ in val_loader:
                sequences = sequences.float().to(device)

                if mask_ratio is not None:
                    if masking_strategy == "frame":
                        masked_inputs, mask = mask_random_frames(sequences, mask_ratio=mask_ratio)
                    elif masking_strategy == "global_joint":
                        joint_dim = 3 if model.three_d == True else 2
                        masked_inputs, mask = mask_random_global_joints(sequences, num_joints=model.num_joints, joint_dim=joint_dim, mask_ratio=mask_ratio)
                    else:
                        raise ValueError(f"Unknown masking strategy: {masking_strategy}")
                else:
                    masked_inputs = sequences
                    mask = torch.ones_like(sequences[..., 0])

                recons = model(masked_inputs)

                loss_matrix = criterion(recons, sequences)

                joint_dim = 3 if model.three_d else 2
                if masking_strategy == "frame":
                    mask_broadcasted = mask.unsqueeze(-1).expand_as(recons)

                elif masking_strategy == "global_joint":
                    mask_broadcasted = mask.repeat_interleave(joint_dim, dim=-1)
                
                masked_loss = loss_matrix * mask_broadcasted
                num_masked = mask_broadcasted.sum()
                loss = masked_loss.sum() / (num_masked + 1e-8)
                val_loss += loss.item() * sequences.size(0)

        val_loss /= len(val_dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model