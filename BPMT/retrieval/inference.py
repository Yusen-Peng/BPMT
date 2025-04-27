import os
import glob
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple
from itertools import combinations
from modality_aware_dataset import GaitRecognitionModalityAwareDataset, PairwiseModalityDataset, finetuningDataset, InferenceDataset
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F
from first_phase_baseline import BaseT1, train_T1
from second_phase_baseline import BaseT2, train_T2, load_T1, CrossAttention
from finetuning import GaitRecognitionHead, finetuning, load_T2, T1_encoding, T2_encoding, load_cross_attn


from utils import set_seed, aggregate_train_val_data_by_camera_split, collect_all_valid_subjects, collate_fn_inference, get_num_joints_for_modality

def evaluate(
    test_loader: torch.utils.data.DataLoader,
    t1_map: dict[str, BaseT1],
    t2_map: dict[str, BaseT2],
    gait_head: GaitRecognitionHead,
    cross_attn_modules_before_T2: dict[str, CrossAttention],
    cross_attn_modules_after_T2: dict[str, CrossAttention],
    device: str = 'cuda'
):
    # freeze all the parameters
    gait_head.eval()
    for m in cross_attn_modules_before_T2.values():
        m.eval()
    for m in cross_attn_modules_after_T2.values():
        m.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            torso_seq, la_seq, ra_seq, ll_seq, rl_seq, labels = batch
            torso_seq = torso_seq.to(device)
            la_seq = la_seq.to(device)
            ra_seq = ra_seq.to(device)
            ll_seq = ll_seq.to(device)
            rl_seq = rl_seq.to(device)
            labels = labels.to(device)

            encoded_t1 = T1_encoding(t1_map, torso_seq, la_seq, ra_seq, ll_seq, rl_seq)
            encoded_T2 = T2_encoding(
                t2_map,
                encoded_t1['torso'],
                encoded_t1['left_arm'],
                encoded_t1['right_arm'],
                encoded_t1['left_leg'],
                encoded_t1['right_leg'],
                cross_attn_modules_before_T2
            )

            def cross_modality_pool(Q, T2_list, module):
                K = V = torch.cat(T2_list, dim=1)
                return module(Q, K, V).mean(dim=1)

            final_repr = torch.cat([
                cross_modality_pool(encoded_t1['torso'], encoded_T2['torso'], cross_attn_modules_after_T2['torso']),
                cross_modality_pool(encoded_t1['left_arm'], encoded_T2['left_arm'], cross_attn_modules_after_T2['left_arm']),
                cross_modality_pool(encoded_t1['right_arm'], encoded_T2['right_arm'], cross_attn_modules_after_T2['right_arm']),
                cross_modality_pool(encoded_t1['left_leg'], encoded_T2['left_leg'], cross_attn_modules_after_T2['left_leg']),
                cross_modality_pool(encoded_t1['right_leg'], encoded_T2['right_leg'], cross_attn_modules_after_T2['right_leg']),
            ], dim=-1)

            logits = gait_head(final_repr)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # metrics
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"[TEST] Accuracy: {acc:.4f}")
    print("[TEST] Classification Report:\n", report)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("figures/test_confusion_matrix.png")
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Gait Recognition Inference")
    parser.add_argument("--root_dir", type=str, default="2D_Poses_50/", help="Root directory of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for Inference")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for the model")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cuda or cpu)")
    return parser.parse_args()


def main():
    set_seed(42)

    args = parse_args()
    root_dir = args.root_dir
    # get the number of classes from the root_dir by taking the trailing number
    batch_size = args.batch_size
    device = args.device

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
    _, _,val_sequences, val_labels = aggregate_train_val_data_by_camera_split(
        valid_subjects,
        train_ratio=0.75,
        seed=42
    )

    # label remapping (IMPORTANT ALL THE TIME!)
    uniqueu_val_labels = sorted(set(val_labels))
    label2new = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(uniqueu_val_labels)}
    val_labels = [label2new[old_lbl] for old_lbl in val_labels]


    # validation/test dataset creation
    torso_val = GaitRecognitionModalityAwareDataset(val_sequences, val_labels, "torso")
    left_arm_val = GaitRecognitionModalityAwareDataset(val_sequences, val_labels, "left_arm")
    right_arm_val = GaitRecognitionModalityAwareDataset(val_sequences, val_labels, "right_arm")
    left_leg_val = GaitRecognitionModalityAwareDataset(val_sequences, val_labels, "left_leg")
    right_leg_val = GaitRecognitionModalityAwareDataset(val_sequences, val_labels, "right_leg")

    # create the test dataset
    test_dataset = InferenceDataset(
        torso_val,
        left_arm_val,
        right_arm_val,
        left_leg_val,
        right_leg_val
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_inference
    )

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

    # load T1 models
    t1_map = {
        'Torso': t1_torso,
        'Left_Arm': t1_left_arm,
        'Right_Arm': t1_right_arm,
        'Left_Leg': t1_left_leg,
        'Right_Leg': t1_right_leg
    }

    
    # load T2 models
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

    
    # load the cross-attention modules before T2
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
        'left_leg_right_leg': cross_attn_ll_rl,
        'right_leg_left_leg': cross_attn_ll_rl,
    }
    

    # load the cross-attention modules after T2
    cross_after_t = load_cross_attn(path="checkpoints/torso_cross_attn_finetuned.pt", d_model=hidden_size, nhead=4, device=device)
    cross_after_la = load_cross_attn(path="checkpoints/left_arm_cross_attn_finetuned.pt", d_model=hidden_size, nhead=4, device=device)
    cross_after_ra = load_cross_attn(path="checkpoints/right_arm_cross_attn_finetuned.pt", d_model=hidden_size, nhead=4, device=device)
    cross_after_ll = load_cross_attn(path="checkpoints/left_leg_cross_attn_finetuned.pt", d_model=hidden_size, nhead=4, device=device)
    cross_after_rl = load_cross_attn(path="checkpoints/right_leg_cross_attn_finetuned.pt", d_model=hidden_size, nhead=4, device=device)

    cross_attn_modules_after_T2 = {
        'torso': cross_after_t,
        'left_arm': cross_after_la,
        'right_arm': cross_after_ra,
        'left_leg': cross_after_ll,
        'right_leg': cross_after_rl,
    }

    # load the gait head
    gait_head = GaitRecognitionHead(input_dim=hidden_size * 5, num_classes=num_classes)
    gait_head.load_state_dict(torch.load("checkpoints/gait_recognition_finetuned.pt", map_location="cpu"))
    gait_head = gait_head.to(device)
    print("Aha! All models loaded successfully!")
    print("=" * 100)


    # evaluate the model
    print("=" * 50)
    print("[INFO] Starting evaluation...")
    print("=" * 50)
    evaluate(
        test_loader,
        t1_map,
        t2_map,
        gait_head,
        cross_attn_modules_before_T2,
        cross_attn_modules_after_T2,
        device=device
    )

    print("=" * 50)
    print("[INFO] Evaluation completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
