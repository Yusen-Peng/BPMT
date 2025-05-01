import os
import glob
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple
from itertools import combinations
from torch import nn
from torch import optim
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
from base_dataset import GaitRecognitionDataset
from utils import set_seed, aggregate_train_val_data_by_camera_split, collect_all_valid_subjects, collate_fn_inference
from finetuning import load_T1, load_T2, load_cross_attn, GaitRecognitionHead

def evaluate(
    data_loader: DataLoader,
    t1: nn.Module,
    t2: nn.Module,
    cross_attn: nn.Module,
    gait_head: nn.Module,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Performs inference and computes accuracy over the given dataset.

    Args:
        data_loader: DataLoader for evaluation
        t1: pretrained (frozen or finetuned) T1 transformer
        t2: trained T2 transformer
        cross_attn: trained CrossAttention module
        gait_head: trained classification head
        device: device to run inference on
        pooling: pooling strategy - 'mean' or 'attention'
        attention_pool: optional attention pooling module (required if pooling == 'attention')

    Returns:
        accuracy: float
        all_preds: tensor of predictions
        all_labels: tensor of ground-truth labels
    """
    t1.eval()
    t2.eval()
    cross_attn.eval()
    gait_head.eval()
   

    all_preds, all_labels = [], []

    with torch.no_grad():
        for skeletons, labels in data_loader:
            skeletons, labels = skeletons.to(device), labels.to(device)

            x1 = t1.encode(skeletons)
            x2 = t2.encode(x1)
            fused = cross_attn(x1, x2, x2)
            pooled = fused.mean(dim=1)

            logits = gait_head(pooled)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)

    print(f"Evaluation Accuracy: {accuracy:.4f}")
    return accuracy, all_preds, all_labels



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
    test_dataset = GaitRecognitionDataset(
        val_sequences,
        val_labels,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_inference
    )

    # load T1 model
    freezeT1 = False
    unfreeze_layers = ["entire"]
    if unfreeze_layers is None:
        t1 = load_T1("baseline_checkpoints/pretrained.pt", d_model=hidden_size, device=device)
    else:
        t1 = load_T1("baseline_checkpoints/finetuned_T1.pt", d_model=hidden_size, device=device)
        print(f"************Unfreezing layers: {unfreeze_layers}")
    
    t2 = load_T2("baseline_checkpoints/finetuned_T2.pt", d_model=hidden_size, device=device)

    # load the cross attention module
    cross_attn = load_cross_attn("baseline_checkpoints/finetuned_cross_attn.pt", d_model=hidden_size, device=device)

    # load the gait recognition head
    gait_head = GaitRecognitionHead(input_dim=hidden_size, num_classes=num_classes)
    gait_head.load_state_dict(torch.load("baseline_checkpoints/finetuned_head.pt", map_location="cpu"))
    gait_head = gait_head.to(device)

    print("Aha! All models loaded successfully!")
    print("=" * 100)

    # evaluate the model
    print("=" * 50)
    print("[INFO] Starting evaluation...")
    print("=" * 50)
    accuracy, all_preds, all_labels = evaluate(
        test_loader,
        t1,
        t2,
        cross_attn,
        gait_head,
        device=device
    )

    print("=" * 50)
    print("[INFO] Evaluation completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
