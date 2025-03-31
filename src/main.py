import os
import glob
import numpy as np
import torch
from typing import List, Tuple
from base_dataset import GaitRecognitionDataset
from modality_aware_dataset import GaitRecognitionModalityAwareDataset
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F
from first_phase_baseline import BaseT1, train_T1

from utils import load_all_data, set_seed, get_num_joints_for_modality


def main():
    set_seed(42)
    root_dir = "2D_Poses_50/"
    batch_size = 4
    num_epochs = 100
    hidden_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 50)
    print(f"[INFO] Starting Gait3D dataset processing on {device}...")
    print("=" * 50)
    sequences, labels = load_all_data(root_dir)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    torso_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "torso")
    left_arm_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "left_arm")
    right_arm_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "right_arm")
    left_leg_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "left_leg")
    right_leg_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "right_leg")

    """"
        First phase masked pretraining: one modality at a time
    """
    modalities = [
            ("Torso", torso_modality),
            ("Left Arm", left_arm_modality),
            ("Right Arm", right_arm_modality),
            ("Left Leg", left_leg_modality),
            ("Right Leg", right_leg_modality),
        ]
    

    for modality_name, modality_dataset in modalities:
        print(f"\n==========================")
        print(f"Starting Masked Pretraining for {modality_name}")
        print(f"==========================")
        
        # figure out how many joints
        num_joints = get_num_joints_for_modality(modality_name)

        # instantiate the model
        model = BaseT1(
            num_joints=num_joints,
            d_model=hidden_size,
            nhead=4,
            num_layers=2
        ).to(device)
        
        # training
        # dataset, model, num_epochs=50, batch_size=16, lr=1e-4, mask_ratio=0.15, device='cuda'):
        train_T1(
            dataset=modality_dataset,
            model=model,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=1e-4,
            mask_ratio=0.15,
            device=device
        )

        # save each model
        torch.save(model.state_dict(), f"{modality_name.lower().replace(' ','_')}_masked_pretrained.pt")

    print("Aha! All modalities trained successfully!")

if __name__ == "__main__":
    main()