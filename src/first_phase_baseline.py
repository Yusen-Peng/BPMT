import torch
import random
import numpy as np
import torch.nn as nn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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



class T1BaseTransformer(nn.Module):
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
        super(T1BaseTransformer, self).__init__()
        self.num_joints = num_joints
        self.d_model = d_model

        # keypoint embedding
        self.embedding = nn.Linear(num_joints * 2, d_model)

        # positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, POSITIONAL_UPPER_BOUND, d_model))

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # reconstruction head
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



if __name__ == "__main__":
    set_seed(42)


    # # CODE TO TEST MASK_KEYPOINTS FUNCTION
    # B = 2
    # T = 5
    # D = 4
    # batch_inputs = torch.rand((B, T, D))
    # mask_ratio = 0.6

    # masked_inputs, mask = mask_keypoints(batch_inputs, mask_ratio)

    # print(f"Original Inputs:\n{batch_inputs}")
    # print(f"Masked Inputs:\n{masked_inputs}")
    # print(f"Mask:\n{mask}")