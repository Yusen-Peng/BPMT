import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import collate_fn_batch_padding
from tqdm import tqdm

POSITIONAL_UPPER_BOUND = 500

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
    
    def __init__(self, num_joints: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super(BaseT1, self).__init__()
        self.num_joints = num_joints
        self.d_model = d_model

        # keypoint embedding
        self.embedding = nn.Linear(num_joints * 2, d_model)

        # positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, POSITIONAL_UPPER_BOUND, d_model))

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # reconstruction head (only used during training)
        self.reconstruction_head = nn.Linear(d_model, num_joints * 2)

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


def train_T1(train_dataset, val_dataset, model, num_epochs=50, batch_size=16, lr=1e-4, mask_ratio=0.15, device='cuda'):
    
    train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size, 
                        shuffle=True,
                        collate_fn=collate_fn_batch_padding
                    )
    val_loader = DataLoader(
                        val_dataset,
                        batch_size=batch_size, 
                        shuffle=False,
                        collate_fn=collate_fn_batch_padding
                    )
    
    # we use MSE loss to measure the reconstruction error
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # we also need to visualize both the train and val loss
    train_losses = []
    val_losses = []

    model.to(device)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        for sequences, _ in train_loader:
            # input sequences: (B, T, 2*num_joints)
            sequences = sequences.float().to(device)

            # perform masking
            #masked_inputs, mask = mask_keypoints(sequences, mask_ratio=mask_ratio)

            # forward pass
            #recons = model(masked_inputs)
            recons = model(sequences)

            # compute the reconstruction loss
            loss = criterion(recons, sequences)
            loss_mean = loss.mean()

            # Check for NaN in loss
            if torch.isnan(loss_mean):
                print("NaN detected in loss, skipping batch")
                continue  # skip this batch instead of crashing

            # we only do MSE on masked positions
            # we also need to broadcast mask to match the shape 
            # mask_broadcasted = mask.unsqueeze(-1).expand_as(recons)
            # masked_loss = loss_matrix * mask_broadcasted

            # compute the average loss per masked position
            #num_masked = mask_broadcasted.sum()
            #loss = masked_loss.sum() / (num_masked + 1e-8)

            # backpropagation
            optimizer.zero_grad()
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # accumulate loss
            train_loss += loss_mean.item() * sequences.size(0)

        # compute the average training loss
        train_loss /= len(train_dataset) 

        # validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, _ in val_loader:
                sequences = sequences.float().to(device)
                #masked_inputs, mask = mask_keypoints(sequences, mask_ratio=mask_ratio)
                #recons = model(masked_inputs)
                recons = model(sequences)

                loss = criterion(recons, sequences)
                loss_mean = loss.mean()
                #mask_broadcasted = mask.unsqueeze(-1).expand_as(recons)
                #masked_loss = loss_matrix * mask_broadcasted
                #num_masked = mask_broadcasted.sum()
                #loss = masked_loss.sum() / (num_masked + 1e-8)
                val_loss += loss_mean.item() * sequences.size(0)

        val_loss /= len(val_dataset)

        # store the losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Val Loss')
    plt.savefig(f'figures/pretrain_val_loss.png')
    
    return model
