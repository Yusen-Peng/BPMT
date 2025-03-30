import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

MASK = False

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