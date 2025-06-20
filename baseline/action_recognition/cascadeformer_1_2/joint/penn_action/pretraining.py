import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

POSITIONAL_UPPER_BOUND = 2048

class SpatialTransformer(nn.Module):
    """
    Applies attention across the joint (spatial) dimension per frame.
    Input shape: (B, T, J, D)
    Output shape: (B, T, J, D)
    """
    def __init__(self, num_joints: int, input_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 1):
        super().__init__()
        self.joint_embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, J, D = x.shape
        x = x.view(B * T, J, D)                   # (B*T, J, D)
        x = self.joint_embedding(x)               # (B*T, J, d_model)
        x = self.encoder(x)                       # (B*T, J, d_model)
        x = self.output_proj(x)                   # (B*T, J, D)
        return x.view(B, T, J, D)                 # (B, T, J, D)


class BaseT1(nn.Module):
    def __init__(self, num_joints: int, three_d: bool, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super(BaseT1, self).__init__()
        self.num_joints = num_joints
        self.input_dim = 3 if three_d else 2
        self.d_model = d_model

        # Spatial transformer over joints
        self.spatial_transformer = SpatialTransformer(num_joints, self.input_dim, d_model=d_model, nhead=nhead)

        # Linear projection to Transformer dimension (J*D → d_model)
        self.joint_embedding = nn.Linear(num_joints * self.input_dim, d_model)

        # Positional encoding over time
        self.pos_embedding = nn.Parameter(torch.zeros(1, POSITIONAL_UPPER_BOUND, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Temporal Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Reconstruction head (d_model → J*D)
        self.reconstruction_head = nn.Linear(d_model, num_joints * self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, D)
        Returns:
            output: (B, T, J, D)
        """
        B, T, J, D = x.shape
        assert D == self.input_dim
        assert T <= POSITIONAL_UPPER_BOUND

        x = self.spatial_transformer(x)                  # (B, T, J, D)
        x = x.view(B, T, J * D)                          # Flatten joints → (B, T, J*D)

        x = self.joint_embedding(x)                      # (B, T, d_model)
        x = x + self.pos_embedding[:, :T, :]             # (B, T, d_model)

        encoded = self.transformer_encoder(x)            # (B, T, d_model)
        decoded = self.reconstruction_head(encoded)      # (B, T, J*D)
        return decoded.view(B, T, J, D)                  # (B, T, J, D)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, J, D = x.shape
        assert D == self.input_dim
        assert T <= POSITIONAL_UPPER_BOUND

        x = self.spatial_transformer(x)                  # (B, T, J, D)
        x = x.view(B, T, J * D)                          # (B, T, J*D)

        x = self.joint_embedding(x)                      # (B, T, d_model)
        x = x + self.pos_embedding[:, :T, :]             # (B, T, d_model)

        return self.transformer_encoder(x)               # (B, T, d_model)

    
PAD_IDX = 0.0

def mask_random_global_joints(inputs: torch.Tensor, mask_ratio: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask joint *coordinates* globally per sample across (T, J),
    and apply mask over all D dimensions of a joint.

    Args:
        inputs: Tensor of shape (B, T, J, D)
        mask_ratio: Fraction of total (T*J) locations to mask

    Returns:
        masked_inputs: Tensor of same shape as inputs with masked values
        mask: Boolean mask of shape (B, T, J) indicating masked joints
    """
    B, T, J, D = inputs.shape
    masked_inputs = inputs.clone()
    mask = torch.zeros(B, T, J, dtype=torch.bool, device=inputs.device)

    total_slots = T * J
    num_to_mask = max(1, int(mask_ratio * total_slots))

    for i in range(B):
        indices = torch.randperm(total_slots, device=inputs.device)[:num_to_mask]
        t_indices = indices // J
        j_indices = indices % J
        mask[i, t_indices, j_indices] = 1

    # Expand to (B, T, J, D)
    expanded_mask = mask.unsqueeze(-1).expand(-1, -1, -1, D)
    masked_inputs[expanded_mask] = PAD_IDX
    return masked_inputs, mask

def train_T1(masking_strategy, train_dataset, val_dataset, model: BaseT1, num_epochs=50, batch_size=16, lr=1e-4, mask_ratio=0.15, device='cuda'):
    from penn_utils import collate_fn_batch_padding

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_batch_padding)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_batch_padding)

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    model.to(device)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        for sequences, _ in train_loader:
            sequences = sequences.float().to(device)  # (B, T, J, D)
            B, T, J, D = sequences.shape

            if masking_strategy == "global_joint":
                masked_inputs, mask = mask_random_global_joints(sequences, mask_ratio=mask_ratio)
                # Expand to (B, T, J, D)
                mask_broadcasted = mask.unsqueeze(-1).expand(B, T, J, D)
            else:
                raise ValueError(f"Unknown masking strategy: {masking_strategy}")

            recons = model(masked_inputs)  # output: (B, T, J, D)
            loss_matrix = criterion(recons, sequences)  # shape: (B, T, J, D)

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
                B, T, J, D = sequences.shape

                masked_inputs, mask = mask_random_global_joints(sequences, mask_ratio=mask_ratio)
                mask_broadcasted = mask.unsqueeze(-1).expand(B, T, J, D)

                recons = model(masked_inputs)
                loss_matrix = criterion(recons, sequences)

                masked_loss = loss_matrix * mask_broadcasted
                num_masked = mask_broadcasted.sum()
                loss = masked_loss.sum() / (num_masked + 1e-8)
                val_loss += loss.item() * sequences.size(0)

        val_loss /= len(val_dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model