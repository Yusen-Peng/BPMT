import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import collate_fn_batch_padding
from first_phase_baseline import BaseT1, mask_keypoints
from second_phase_baseline import BaseT2, CrossAttention, load_T1


def load_T2(model_path: str, out_dim_A: int, out_dim_B: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, freeze: bool = True,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> BaseT2:
    """
        loads a BaseT2 model from a checkpoint
    """
    model = BaseT2(out_dim_A=out_dim_A, out_dim_B=out_dim_B, d_model=d_model, nhead=nhead, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # optionally freeze the model parameters
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        
    # move model to device and return the model
    return model.to(device)



class GaitRecognitionHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # a simple linear classifier
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, input_dim)
        return self.fc(x)










