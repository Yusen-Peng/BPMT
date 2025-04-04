import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import collate_fn_batch_padding

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


def train_T1(dataset, model, num_epochs=50, batch_size=16, lr=1e-4, mask_ratio=0.15, device='cuda'):
    """
        Train the transformer model on the dataset.
        
        Args:
            dataset: The dataset containing sequences and labels.
            model: The transformer model.
            num_epochs: Number of training epochs.
            batch_size: Batch size for training.
            lr: Learning rate for the optimizer.
            mask_ratio: Fraction of frames to mask during training.
            device: Device to run the training on ('cuda' or 'cpu').
    """

    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size, 
                        shuffle=True,
                        collate_fn=collate_fn_batch_padding
                    )
    
    # we use MSE loss to measure the reconstruction error
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for sequences, _ in dataloader:
            # input sequences: (B, T, 2*num_joints)
            sequences = sequences.float().to(device)

            # perform masking
            masked_inputs, mask = mask_keypoints(sequences, mask_ratio=mask_ratio)

            # forward pass
            recons = model(masked_inputs)

            # compute the reconstruction loss
            loss_matrix = criterion(recons, sequences)

            # we only do MSE on masked positions
            # we also need to broadcast mask to match the shape 
            mask_broadcasted = mask.unsqueeze(-1).expand_as(recons)
            masked_loss = loss_matrix * mask_broadcasted

            # compute the average loss per masked position
            num_masked = mask_broadcasted.sum()
            loss = masked_loss.sum() / (num_masked + 1e-8)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate loss
            epoch_loss += loss.item() * sequences.size(0)

        # average epoch loss
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    return model
