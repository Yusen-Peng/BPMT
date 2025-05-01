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
from finetuning import load_T1, load_T2, load_cross_attn

def evaluate(
    data_loader: DataLoader,
    t1: nn.Module,
    t2: nn.Module,
    cross_attn: nn.Module,
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

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for skeletons, labels in data_loader:
            skeletons, labels = skeletons.to(device), labels.to(device)
            x1 = t1.encode(skeletons)
            x2 = t2.encode(x1)
            fused = cross_attn(x1, x2, x2)

            # pooling
            pooled = fused.mean(dim=1)

            all_embeddings.append(pooled)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute pairwise distances
    distance_matrix = torch.cdist(all_embeddings, all_embeddings, p=2) 

    # for each sample, rank others by distance
    ranks = distance_matrix.argsort(dim=1)

    correct = 0
    total = all_labels.size(0)

    for i in range(total):
        # we need to skip the scenario where the sample is the same as itself
        for j in ranks[i]:
            if j == i:
                continue
            
            if all_labels[j] == all_labels[i]:
                # top-1 is correct
                correct += 1
                break 
            else:
                # top-1 is not correct
                correct += 0
                break

    rank1_accuracy = correct / total

    return rank1_accuracy, all_embeddings, all_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Gait Recognition Inference")
    parser.add_argument("--root_dir", type=str, default="2D_Poses_50_unseen/", help="Root directory of the dataset")
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
    print("=" * 100)
    print(f"[INFO] Number of valid subjects: {num_classes}")
    print("=" * 100)


    # split the dataset into training and validation sets
    train_sequences, train_labels,val_sequences, val_labels = aggregate_train_val_data_by_camera_split(
        valid_subjects,
        train_ratio=0.75,
        seed=42
    )

    # label remapping (IMPORTANT ALL THE TIME!)
    # for unseen inference, we need to remap the labels to be continuous
    # get the unique labels in the training set
    uniqueu_train_labels = sorted(set(train_labels))
    label2new = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(uniqueu_train_labels)}
    train_labels = [label2new[old_lbl] for old_lbl in train_labels]
    uniqueu_val_labels = sorted(set(val_labels))
    label2new = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(uniqueu_val_labels)}
    val_labels = [label2new[old_lbl] for old_lbl in val_labels]


    # merge train and validation sets
    all_sequences = train_sequences + val_sequences
    all_labels = train_labels + val_labels

    # validation/test dataset creation
    test_dataset = GaitRecognitionDataset(
        all_sequences,
        all_labels,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_inference
    )

    # load T1 model
    unfreeze_layers = [1]
    if unfreeze_layers is None:
        t1 = load_T1("baseline_checkpoints/pretrained.pt", d_model=hidden_size, device=device)
    else:
        t1 = load_T1("baseline_checkpoints/finetuned_T1.pt", d_model=hidden_size, device=device)
        print(f"************Unfreezing layers: {unfreeze_layers}")

    
    t2 = load_T2("baseline_checkpoints/finetuned_T2.pt", d_model=hidden_size, device=device)

    # load the cross attention module
    cross_attn = load_cross_attn("baseline_checkpoints/finetuned_cross_attn.pt", d_model=hidden_size, device=device)

    print("Aha! All models loaded successfully!")
    print("=" * 100)

    # evaluate the model
    print("=" * 50)
    print("[INFO] Starting evaluation...")
    rank1_accuracy, all_preds, all_labels = evaluate(
        test_loader,
        t1,
        t2,
        cross_attn,
        device=device
    )

    print("[INFO] Evaluation completed!")
    print(f"[INFO] Rank-1 accuracy: {rank1_accuracy:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
