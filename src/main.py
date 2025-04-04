import os
import glob
import numpy as np
import torch
from typing import List, Tuple
from itertools import combinations
from modality_aware_dataset import GaitRecognitionModalityAwareDataset, PairwiseModalityDataset, finetuningDataset
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F
from first_phase_baseline import BaseT1, train_T1
from second_phase_baseline import BaseT2, train_T2, load_T1
from finetuning import GaitRecognitionHead, finetuning, load_T2


from utils import load_all_data, set_seed, get_num_joints_for_modality


def main():
    set_seed(42)
    # base experiment
    root_dir = "2D_Poses_50/"
    # tiny experiment
    #root_dir = "2D_Poses_300/"

    # full experiment
    #root_dir = "2D_Poses_1000/"

    batch_size = 4
    num_epochs = 100
    hidden_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print("=" * 50)
    # print(f"[INFO] Starting Gait3D dataset processing on {device}...")
    # print("=" * 50)
    # sequences, labels = load_all_data(root_dir)
    # torso_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "torso")
    # left_arm_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "left_arm")
    # right_arm_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "right_arm")
    # left_leg_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "left_leg")
    # right_leg_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "right_leg")

    # """"
    #     First phase masked pretraining: one modality at a time
    # """
    # modalities = [
    #         ("Torso", torso_modality),
    #         ("Left_Arm", left_arm_modality),
    #         ("Right_Arm", right_arm_modality),
    #         ("Left_Leg", left_leg_modality),
    #         ("Right_Leg", right_leg_modality),
    #     ]
    


    # for modality_name, modality_dataset in modalities:
    #     print(f"\n==========================")
    #     print(f"Starting Masked Pretraining for {modality_name}")
    #     print(f"==========================")
        
    #     # figure out how many joints
    #     num_joints = get_num_joints_for_modality(modality_name)

    #     # instantiate the model
    #     model = BaseT1(
    #         num_joints=num_joints,
    #         d_model=hidden_size,
    #         nhead=4,
    #         num_layers=2
    #     ).to(device)
        
    #     # training
    #     # dataset, model, num_epochs=50, batch_size=16, lr=1e-4, mask_ratio=0.15, device='cuda'):
    #     train_T1(
    #         dataset=modality_dataset,
    #         model=model,
    #         num_epochs=num_epochs,
    #         batch_size=batch_size,
    #         lr=1e-4,
    #         mask_ratio=0.15,
    #         device=device
    #     )

    #     # save each model
    #     torch.save(model.state_dict(), f"checkpoints/{modality_name.lower().replace(' ','_')}_masked_pretrained.pt")

    # print("Aha! All single modalities trained successfully!")
    # print("=" * 100)
    # print("=" * 100)
    # print("=" * 100)
    # print("=" * 100)

    # """
    #     Second phase masked pretraining: one pair of modalities at a time
    # """

    # modality_map = dict(modalities)
    # modality_names = list(modality_map.keys())

    # for modA_name, modB_name in combinations(modality_names, 2):
    #     print(f"\n==========================")
    #     print(f"Second-Stage Pretraining on {modA_name} + {modB_name}")
    #     print(f"==========================")

    #     datasetA = modality_map[modA_name]
    #     datasetB = modality_map[modB_name]
    #     pairwise_dataset = PairwiseModalityDataset(datasetA, datasetB)

    #     num_joints_A = get_num_joints_for_modality(modA_name)
    #     num_joints_B = get_num_joints_for_modality(modB_name)

    #     model_T2 = train_T2(
    #         pairwise_dataset=pairwise_dataset,
    #         model_pathA=f"checkpoints/{modA_name.lower().replace(' ','_')}_masked_pretrained.pt",
    #         model_pathB=f"checkpoints/{modB_name.lower().replace(' ','_')}_masked_pretrained.pt",
    #         num_joints_A=num_joints_A,
    #         num_joints_B=num_joints_B,
    #         d_model=hidden_size,
    #         nhead=4,
    #         num_layers=2,
    #         num_epochs=num_epochs,
    #         batch_size=batch_size,
    #         lr=1e-4,
    #         mask_ratio=0.15,
    #         freeze_T1=True,
    #         device=device
    #     )

    #     save_path = f"checkpoints/{modA_name}_{modB_name}_T2.pt"
    #     torch.save(model_T2.state_dict(), save_path)

    # print("Aha! All modality pairs trained successfully!")
    # print("=" * 100)
    

    """
        finetuning on gait recognition.
    """
    num_joints_dict = {
        "Torso": get_num_joints_for_modality("Torso"),
        "Left_Arm": get_num_joints_for_modality("Left_Arm"),
        "Right_Arm": get_num_joints_for_modality("Right_Arm"),
        "Left_Leg": get_num_joints_for_modality("Left_Leg"),
        "Right_Leg": get_num_joints_for_modality("Right_Leg"),    
    }


    # load T1 models
    #load_T1(model_path: str, num_joints: int = 14, d_model: int = 128, nhead: int = 4, num_layers: int = 2, freeze: bool = True,
            #device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> BaseT1:
    t1_torso = load_T1(
        model_path="checkpoints/torso_masked_pretrained.pt",
        num_joints=num_joints_dict["Torso"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t1_left_arm = load_T1(
        model_path="checkpoints/left_arm_masked_pretrained.pt",
        num_joints=num_joints_dict["Left_Arm"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t1_right_arm = load_T1(
        model_path="checkpoints/right_arm_masked_pretrained.pt",
        num_joints=num_joints_dict["Right_Arm"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t1_left_leg = load_T1(
        model_path="checkpoints/left_leg_masked_pretrained.pt",
        num_joints=num_joints_dict["Left_Leg"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t1_right_leg = load_T1(
        model_path="checkpoints/right_leg_masked_pretrained.pt",
        num_joints=num_joints_dict["Right_Leg"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )

    # load T2 models
    #load_T2(model_path: str, out_dim_A: int, out_dim_B: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, freeze: bool = True,
            #device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> BaseT2:
    t2_torso_left_arm = load_T2(
        model_path="checkpoints/Torso_Left_Arm_T2.pt",
        out_dim_A=num_joints_dict["Torso"],
        out_dim_B=num_joints_dict["Left_Arm"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_torso_right_arm = load_T2(
        model_path="checkpoints/Torso_Right_Arm_T2.pt",
        out_dim_A=num_joints_dict["Torso"],
        out_dim_B=num_joints_dict["Right_Arm"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_torso_left_leg = load_T2(
        model_path="checkpoints/Torso_Left_Leg_T2.pt",
        out_dim_A=num_joints_dict["Torso"],
        out_dim_B=num_joints_dict["Left_Leg"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_torso_right_leg = load_T2(
        model_path="checkpoints/Torso_Right_Leg_T2.pt",
        out_dim_A=num_joints_dict["Torso"],
        out_dim_B=num_joints_dict["Right_Leg"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_left_arm_right_arm = load_T2(
        model_path="checkpoints/Left_Arm_Right_Arm_T2.pt",
        out_dim_A=num_joints_dict["Left_Arm"],
        out_dim_B=num_joints_dict["Right_Arm"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_left_arm_left_leg = load_T2(
        model_path="checkpoints/Left_Arm_Left_Leg_T2.pt",
        out_dim_A=num_joints_dict["Left_Arm"],
        out_dim_B=num_joints_dict["Left_Leg"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_left_arm_right_leg = load_T2(
        model_path="checkpoints/Left_Arm_Right_Leg_T2.pt",
        out_dim_A=num_joints_dict["Left_Arm"],
        out_dim_B=num_joints_dict["Right_Leg"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_right_arm_left_leg = load_T2(
        model_path="checkpoints/Right_Arm_Left_Leg_T2.pt",
        out_dim_A=num_joints_dict["Right_Arm"],
        out_dim_B=num_joints_dict["Left_Leg"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_right_arm_right_leg = load_T2(
        model_path="checkpoints/Right_Arm_Right_Leg_T2.pt",
        out_dim_A=num_joints_dict["right_arm"],
        out_dim_B=num_joints_dict["right_leg"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_left_leg_right_leg = load_T2(
        model_path="checkpoints/Left_Leg_Right_Leg_T2.pt",
        out_dim_A=num_joints_dict["left_leg"],
        out_dim_B=num_joints_dict["right_leg"],
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )

    
    t1_map = {
        'torso': t1_torso,
        'left_arm': t1_left_arm,
        'right_arm': t1_right_arm,
        'left_leg': t1_left_leg,
        'right_leg': t1_right_leg
    }

    t2_map = {

        'torso_left_arm': t2_torso_left_arm,
        'torso_right_arm': t2_torso_right_arm,
        'torso_left_leg': t2_torso_left_leg,
        'torso_right_leg': t2_torso_right_leg,
        
        'left_arm_right_arm': t2_left_arm_right_arm,
        'left_arm_left_leg': t2_left_arm_left_leg,
        'left_arm_right_leg': t2_left_arm_right_leg,

        'right_arm_left_leg': t2_right_arm_left_leg,
        'right_arm_right_leg': t2_right_arm_right_leg,


        'left_leg_right_leg': t2_left_leg_right_leg,
    }

    print("Aha! All models loaded successfully!")
    print("=" * 100)





if __name__ == "__main__":
    main()