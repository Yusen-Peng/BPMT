import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import collate_fn_batch_padding
from tqdm import tqdm

MASK = False
POSITIONAL_UPPER_BOUND = 500

def mask_keypoints(batch_inputs: torch.Tensor, mask_ratio: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly masks a fraction of frames in the input batch.
    Args:
    batch_inputs: shape (B, T, D)
       B = batch size
       T = #frames
       D = dimension of each frame:  2 * #joints for one specific modality (x,y)

    mask_ratio: fraction of frames to mask

    returns:
      masked_inputs: same shape as batch_inputs, with masked frames replaced by [MASK]
      mask: boolean mask of shape (B, T), where True indicates masked positions.
    """

    # destructure the input tensor
    B, T, D = batch_inputs.shape

    # Number of frames to mask per sequence
    num_to_mask = int(T * mask_ratio)

    # create the mask tensor
    mask = torch.zeros((B, T), dtype=torch.bool, device=batch_inputs.device)
    masked_inputs = batch_inputs.clone()

    for i in range(B):
        # randomly select frames to mask
        mask_indices = torch.randperm(T)[:num_to_mask]
        mask[i, mask_indices] = True

        # perform masking
        masked_inputs[i, mask_indices, :] = MASK

    return masked_inputs, mask


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
            Encode the input sequence without reconstruction.
            This is used for extracting features from the model.
        """
        B, T, _ = x.shape
        keypoint_embedding = self.embedding(x)
        keypoint_embedding_with_pos = keypoint_embedding + self.pos_embedding[:, :T, :]

        # NOTE: PyTorch Transformer wants shape (T, B, d_model) instead of (B, T, d_model)
        keypoint_embedding_with_pos = keypoint_embedding_with_pos.transpose(0,1)
        encoded = self.transformer_encoder(keypoint_embedding_with_pos)
        encoded = encoded.transpose(0,1)

        return encoded


def train_T1(modality_name, train_dataset, val_dataset, model, num_epochs=50, batch_size=16, lr=1e-4, mask_ratio=0.15, device='cuda'):
    
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
    plt.title(f'{modality_name} - Train and Val Loss')
    plt.savefig(f'figures/{modality_name}_train_val_loss.png')
    
    return model
