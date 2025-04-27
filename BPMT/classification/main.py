import os
import glob
import numpy as np
import torch
import argparse
from typing import List, Tuple
from itertools import combinations
from modality_aware_dataset import GaitRecognitionModalityAwareDataset, PairwiseModalityDataset, finetuningDataset
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F
from first_phase_baseline import BaseT1, train_T1
from second_phase_baseline import BaseT2, train_T2, load_T1
from finetuning import GaitRecognitionHead, finetuning, load_T2, load_cross_attn


from utils import set_seed, get_num_joints_for_modality, collate_fn_finetuning, aggregate_train_val_data_by_camera_split, collect_all_valid_subjects


def parse_args():
    parser = argparse.ArgumentParser(description="Gait Recognition Training")
    parser.add_argument("--first_stage", action='store_true', help="Run the first stage of pretraining")
    parser.add_argument("--second_stage", action='store_true', help="Run the second stage of pretraining")
    parser.add_argument("--root_dir", type=str, default="2D_Poses_50/", help="Root directory of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for the model")
    parser.add_argument("--class_specific_split", action='store_true', help="Use class-specific split for training and validation")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cuda or cpu)")
    return parser.parse_args()


def main():
    set_seed(42)

    args = parse_args()
    root_dir = args.root_dir
    # get the number of classes from the root_dir by taking the trailing number
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size
    device = args.device
    first_stage = args.first_stage
    second_stage = args.second_stage
    class_specific_split = args.class_specific_split

    print(f"first_stage: {first_stage}")
    print(f"second_stage: {second_stage}")
    print(f"using class-specific split: {class_specific_split}")

    # Set the device

    hidden_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 50)
    print(f"[INFO] Starting Gait3D dataset processing on {device}...")
    print("=" * 50)

    MIN_CAMERAS = 3

    # load the dataset
    valid_subjects = collect_all_valid_subjects(root_dir, min_cameras=MIN_CAMERAS)

    # get the number of classes
    num_classes = len(valid_subjects)
    print(f"[INFO] Number of classes: {num_classes}")
    print("=" * 100)


    # split the dataset into training and validation sets
    train_sequences, train_labels, val_sequences, val_labels = aggregate_train_val_data_by_camera_split(
        valid_subjects,
        train_ratio=0.75,
        seed=42
    )


    # label remapping (IMPORTANT ALL THE TIME!)
    unique_train_labels = sorted(set(train_labels))
    label2new = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(unique_train_labels)}
    train_labels = [label2new[old_lbl] for old_lbl in train_labels]
    
    uniqueu_val_labels = sorted(set(val_labels))
    label2new = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(uniqueu_val_labels)}
    val_labels = [label2new[old_lbl] for old_lbl in val_labels]


    # dataset creation
    torso_train = GaitRecognitionModalityAwareDataset(train_sequences, train_labels, "torso")
    torso_val = GaitRecognitionModalityAwareDataset(val_sequences, val_labels, "torso")
    left_arm_train = GaitRecognitionModalityAwareDataset(train_sequences, train_labels, "left_arm")
    left_arm_val = GaitRecognitionModalityAwareDataset(val_sequences, val_labels, "left_arm")
    right_arm_train = GaitRecognitionModalityAwareDataset(train_sequences, train_labels, "right_arm")
    right_arm_val = GaitRecognitionModalityAwareDataset(val_sequences, val_labels, "right_arm")
    left_leg_train = GaitRecognitionModalityAwareDataset(train_sequences, train_labels, "left_leg")
    left_leg_val = GaitRecognitionModalityAwareDataset(val_sequences, val_labels, "left_leg")
    right_leg_train = GaitRecognitionModalityAwareDataset(train_sequences, train_labels, "right_leg")
    right_leg_val = GaitRecognitionModalityAwareDataset(val_sequences, val_labels, "right_leg")

    # define modalities
    modalities = [
        ("Torso", torso_train, torso_val),
        ("Left_Arm", left_arm_train, left_arm_val),
        ("Right_Arm", right_arm_train, right_arm_val),
        ("Left_Leg", left_leg_train, left_leg_val),
        ("Right_Leg", right_leg_train, right_leg_val)
    ]

    if first_stage == True: 
        """"
            First phase masked pretraining: one modality at a time
        """

        for modality_name, train_dataset, val_dataset in modalities: 
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
                modality_name=modality_name,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
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
    
    
    print("=" * 100)
    print("=" * 100)
    print("=" * 100)


    modality_map = {
        "Torso": (torso_train, torso_val),
        "Left_Arm": (left_arm_train, left_arm_val),
        "Right_Arm": (right_arm_train, right_arm_val),
        "Left_Leg": (left_leg_train, left_leg_val),
        "Right_Leg": (right_leg_train, right_leg_val)
    }


    if second_stage == True:
        """
            Second phase masked pretraining: one pair of modalities at a time
        """

        # define the modality names from the dictionary
        modality_names = list(modality_map.keys())

        for modA_name, modB_name in combinations(modality_names, 2):
            print(f"\n==========================")
            print(f"Second-Stage Pretraining on {modA_name} + {modB_name}")
            print(f"==========================")


            datasetA_train, datasetA_val = modality_map[modA_name]
            datasetB_train, datasetB_val = modality_map[modB_name]

            train_pairwise_dataset = PairwiseModalityDataset(datasetA_train, datasetB_train)
            val_pairwise_dataset = PairwiseModalityDataset(datasetA_val, datasetB_val)

            num_joints_A = get_num_joints_for_modality(modA_name)
            num_joints_B = get_num_joints_for_modality(modB_name)

            model_T2, cross_attn = train_T2(
                modality_name_A=modA_name,
                modality_name_B=modB_name,
                train_pairwise_dataset=train_pairwise_dataset,
                val_pairwise_dataset=val_pairwise_dataset,
                model_pathA=f"checkpoints/{modA_name.lower().replace(' ','_')}_masked_pretrained.pt",
                model_pathB=f"checkpoints/{modB_name.lower().replace(' ','_')}_masked_pretrained.pt",
                num_joints_A=num_joints_A,
                num_joints_B=num_joints_B,
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

            # also save each cross-attentions
            cross_attn_save_path = f"checkpoints/{modA_name}_{modB_name}_cross_attn.pt"
            torch.save(cross_attn.state_dict(), cross_attn_save_path)

        print("Aha! All modality pairs trained successfully!")
        print("=" * 100)

    
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

    t2_torso_left_arm = load_T2(
        model_path="checkpoints/Torso_Left_Arm_T2.pt",
        out_dim_A=num_joints_dict["Torso"] * 2,
        out_dim_B=num_joints_dict["Left_Arm"] * 2,
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_torso_right_arm = load_T2(
        model_path="checkpoints/Torso_Right_Arm_T2.pt",
        out_dim_A=num_joints_dict["Torso"] * 2,
        out_dim_B=num_joints_dict["Right_Arm"] * 2,
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_torso_left_leg = load_T2(
        model_path="checkpoints/Torso_Left_Leg_T2.pt",
        out_dim_A=num_joints_dict["Torso"] * 2,
        out_dim_B=num_joints_dict["Left_Leg"] * 2,
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_torso_right_leg = load_T2(
        model_path="checkpoints/Torso_Right_Leg_T2.pt",
        out_dim_A=num_joints_dict["Torso"] * 2,
        out_dim_B=num_joints_dict["Right_Leg"] * 2,
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_left_arm_right_arm = load_T2(
        model_path="checkpoints/Left_Arm_Right_Arm_T2.pt",
        out_dim_A=num_joints_dict["Left_Arm"] * 2,
        out_dim_B=num_joints_dict["Right_Arm"] * 2,
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_left_arm_left_leg = load_T2(
        model_path="checkpoints/Left_Arm_Left_Leg_T2.pt",
        out_dim_A=num_joints_dict["Left_Arm"] * 2,
        out_dim_B=num_joints_dict["Left_Leg"] * 2,
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_left_arm_right_leg = load_T2(
        model_path="checkpoints/Left_Arm_Right_Leg_T2.pt",
        out_dim_A=num_joints_dict["Left_Arm"] * 2,
        out_dim_B=num_joints_dict["Right_Leg"] * 2,
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_right_arm_left_leg = load_T2(
        model_path="checkpoints/Right_Arm_Left_Leg_T2.pt",
        out_dim_A=num_joints_dict["Right_Arm"] * 2,
        out_dim_B=num_joints_dict["Left_Leg"] * 2,
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_right_arm_right_leg = load_T2(
        model_path="checkpoints/Right_Arm_Right_Leg_T2.pt",
        out_dim_A=num_joints_dict["Right_Arm"] * 2,
        out_dim_B=num_joints_dict["Right_Leg"] * 2,
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )
    t2_left_leg_right_leg = load_T2(
        model_path="checkpoints/Left_Leg_Right_Leg_T2.pt",
        out_dim_A=num_joints_dict["Left_Leg"] * 2,
        out_dim_B=num_joints_dict["Right_Leg"] * 2,
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )

    # also load the cross-attention models before T2 and construct the map
    cross_attn_t_la = load_cross_attn(path="checkpoints/Torso_Left_Arm_cross_attn.pt", d_model=hidden_size, nhead=4, device=device)
    cross_attn_t_ra = load_cross_attn(path="checkpoints/Torso_Right_Arm_cross_attn.pt", d_model=hidden_size, nhead=4, device=device)
    cross_attn_t_ll = load_cross_attn(path="checkpoints/Torso_Left_Leg_cross_attn.pt", d_model=hidden_size, nhead=4, device=device)
    cross_attn_t_rl = load_cross_attn(path="checkpoints/Torso_Right_Leg_cross_attn.pt", d_model=hidden_size, nhead=4, device=device)
    cross_attn_la_ra = load_cross_attn(path="checkpoints/Left_Arm_Right_Arm_cross_attn.pt", d_model=hidden_size, nhead=4, device=device)
    cross_attn_la_ll = load_cross_attn(path="checkpoints/Left_Arm_Left_Leg_cross_attn.pt", d_model=hidden_size, nhead=4, device=device)
    cross_attn_la_rl = load_cross_attn(path="checkpoints/Left_Arm_Right_Leg_cross_attn.pt", d_model=hidden_size, nhead=4, device=device)
    cross_attn_ra_ll = load_cross_attn(path="checkpoints/Right_Arm_Left_Leg_cross_attn.pt", d_model=hidden_size, nhead=4, device=device)
    cross_attn_ra_rl = load_cross_attn(path="checkpoints/Right_Arm_Right_Leg_cross_attn.pt", d_model=hidden_size, nhead=4, device=device)
    cross_attn_ll_rl = load_cross_attn(path="checkpoints/Left_Leg_Right_Leg_cross_attn.pt", d_model=hidden_size, nhead=4, device=device)

    # cross-attention modules before T2
    cross_attn_modules_before_T2 = {
        'torso_left_arm': cross_attn_t_la,
        'torso_right_arm': cross_attn_t_ra,
        'torso_left_leg': cross_attn_t_ll,
        'torso_right_leg': cross_attn_t_rl,
        'left_arm_right_arm': cross_attn_la_ra,
        'left_arm_left_leg': cross_attn_la_ll,
        'left_arm_right_leg': cross_attn_la_rl,
        'right_arm_left_leg': cross_attn_ra_ll,
        'right_arm_right_leg': cross_attn_ra_rl,
        'left_leg_right_leg': cross_attn_ll_rl
    }

    t1_map = {
        'Torso': t1_torso,
        'Left_Arm': t1_left_arm,
        'Right_Arm': t1_right_arm,
        'Left_Leg': t1_left_leg,
        'Right_Leg': t1_right_leg
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


    train_finetuning_dataset = finetuningDataset(
        torso_dataset=torso_train,
        left_arm_dataset=left_arm_train,
        right_arm_dataset=right_arm_train,
        left_leg_dataset=left_leg_train,
        right_leg_dataset=right_leg_train,
    )

    train_finetuning_dataloader = torch.utils.data.DataLoader(
        train_finetuning_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_finetuning
    )

    val_finetuning_dataset = finetuningDataset(
        torso_dataset=torso_val,
        left_arm_dataset=left_arm_val,
        right_arm_dataset=right_arm_val,
        left_leg_dataset=left_leg_val,
        right_leg_dataset=right_leg_val,
    )

    val_finetuning_dataloader = torch.utils.data.DataLoader(
        val_finetuning_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_finetuning
    )

    gait_head_template = GaitRecognitionHead(input_dim=hidden_size * 5, num_classes=num_classes).to(device)

    gait_head, cross_attn_modules_after_T2 = finetuning(
        train_loader=train_finetuning_dataloader,
        val_loader=val_finetuning_dataloader,
        t1_map=t1_map,
        t2_map=t2_map,
        cross_attn_modules_before_T2=cross_attn_modules_before_T2,
        gait_head=gait_head_template,
        d_model=hidden_size,
        num_epochs=num_epochs,
        freeze=False,   # change this to False if we want to finetune the T1 and T2 models
        device=device
    )

    print("Aha! Finetuning completed successfully!")


    # save the finetuned model
    torch.save(gait_head.state_dict(), "checkpoints/gait_recognition_finetuned.pt")
    # save the cross-attention modules after T2
    for key, value in cross_attn_modules_after_T2.items():
        torch.save(value.state_dict(), f"checkpoints/{key}_cross_attn_finetuned.pt")

if __name__ == "__main__":
    main()