import os
import glob
import numpy as np
import torch
from typing import List, Tuple
from itertools import combinations
from modality_aware_dataset import GaitRecognitionModalityAwareDataset, PairwiseModalityDataset
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F
from first_phase_baseline import BaseT1, train_T1
from second_phase_baseline import BaseT2, train_T2


from utils import load_all_data, set_seed, get_num_joints_for_modality


def main():
    set_seed(42)
    #root_dir = "2D_Poses_50/"
    # scale up to 300 now
    root_dir = "2D_Poses_300/"

    batch_size = 4
    num_epochs = 100
    hidden_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 50)
    print(f"[INFO] Starting Gait3D dataset processing on {device}...")
    print("=" * 50)
    sequences, labels = load_all_data(root_dir)
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
        torch.save(model.state_dict(), f"checkpoints/{modality_name.lower().replace(' ','_')}_masked_pretrained.pt")

    print("Aha! All single modalities trained successfully!")
    print("=" * 100)

    """
        Second phase masked pretraining: one pair of modalities at a time
    """

    modality_map = dict(modalities)
    modality_names = list(modality_map.keys())

    for modA_name, modB_name in combinations(modality_names, 2):
        print(f"\n==========================")
        print(f"Second-Stage Pretraining on {modA_name} + {modB_name}")
        print(f"==========================")

        datasetA = modality_map[modA_name]
        datasetB = modality_map[modB_name]
        pairwise_dataset = PairwiseModalityDataset(datasetA, datasetB)

        num_joints_A = get_num_joints_for_modality(modA_name)
        num_joints_B = get_num_joints_for_modality(modB_name)

        model_T2 = train_T2(
            pairwise_dataset=pairwise_dataset,
            model_pathA=f"checkpoints/{modA_name.lower().replace(' ','_')}_masked_pretrained.pt",
            model_pathB=f"checkpoints/{modB_name.lower().replace(' ','_')}_masked_pretrained.pt",
            num_joints=(num_joints_A, num_joints_B),
            d_model=hidden_size,
            nhead=4,
            num_layers=2,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=1e-4,
            mask_ratio=0.15,
            freeze_T1=True,
            device=device
        )

        save_path = f"checkpoints/{modA_name}_{modB_name}_T2.pt"
        torch.save(model_T2.state_dict(), save_path)

        print("Aha! All modality pairs trained successfully!")
        print("=" * 100)


if __name__ == "__main__":
    main()