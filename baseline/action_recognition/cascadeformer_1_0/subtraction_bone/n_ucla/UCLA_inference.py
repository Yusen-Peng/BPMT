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
from base_dataset import ActionRecognitionDataset
from finetuning import load_T1, load_T2, load_cross_attn, GaitRecognitionHead
from UCLA_utils import set_seed, build_nucla_action_lists_cross_view, split_train_val, collate_fn_inference, NUM_JOINTS_NUCLA
from SF_UCLA_loader import SF_UCLA_Dataset, skateformer_collate_fn

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

    return accuracy, all_preds, all_labels

def parse_args():
    parser = argparse.ArgumentParser(description="Gait Recognition Inference")
    parser.add_argument("--root_dir", type=str, default="N_UCLA/", help="Root directory of the dataset")
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

    hidden_size = 256
    n_heads = 8
    num_layers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 50)
    print(f"[INFO] Starting N-UCLA dataset processing on {device}...")
    print("=" * 50)

    # # load the dataset
    # train_seq, train_lbl, test_seq, test_lbl = build_nucla_action_lists_cross_view(
    #     root=root_dir,
    #     train_views=['view_1', 'view_2'],
    #     test_views=['view_3']
    # )
    # train_seq, train_lbl, val_seq, val_lbl = split_train_val(train_seq, train_lbl, val_ratio=0.05)
    
    # test_dataset = ActionRecognitionDataset(test_seq, test_lbl)

    train_data_path = 'N-UCLA_processed/'
    train_label_path = 'N-UCLA_processed/train_label.pkl'


    data_type = 'b'
    train_dataset_pre = SF_UCLA_Dataset(
        data_path=train_data_path,
        label_path=train_label_path,
        data_type=data_type, 
        window_size=64, 
        partition=True, 
        repeat=1, 
        p=0.5, 
        debug=False
    )

    train_seq = []
    train_lbl = []

    for i in range(len(train_dataset_pre)):
        data, _, label, _ = train_dataset_pre[i]
        # FIXME: a better reshape strategy
        data_tensor = torch.from_numpy(data).permute(1, 0, 2, 3).reshape(data.shape[1], -1)
        train_seq.append(data_tensor)
        train_lbl.append(label)

    print(f"Collected {len(train_seq)} sequences for train + val.")
    print(f"Each sequence shape: {train_seq[0].shape}")  # (64, 60)

    # train-val split
    val_ratio = 0.05
    train_seq, train_lbl, val_seq, val_lbl = split_train_val(train_seq, train_lbl, val_ratio=val_ratio)

    test_data_path = 'N-UCLA_processed/'
    test_label_path = 'N-UCLA_processed/val_label.pkl'

    P_INFERENCE_MODE = 0.0
    REPEAT_INFERENCE_MODE = 1
    test_dataset_pre = SF_UCLA_Dataset(
        data_path=test_data_path,
        label_path=test_label_path,
        data_type=data_type, 
        window_size=-1, 
        partition=True, 
        repeat=REPEAT_INFERENCE_MODE, 
        p=P_INFERENCE_MODE, 
        debug=False
    )

    test_seq = []
    test_lbl = []
    for i in range(len(test_dataset_pre)):
        data, _, label, _ = test_dataset_pre[i]
        data_tensor = torch.from_numpy(data).permute(1, 0, 2, 3).reshape(data.shape[1], -1)
        test_seq.append(data_tensor)
        test_lbl.append(label)
    
    print(f"Collected {len(test_seq)} sequences for test.")
    print(f"Each sequence shape: {test_seq[0].shape}")  # (64, 60)
    
    # get the number of classes
    num_classes = max(train_lbl + val_lbl + test_lbl) + 1
    print(f"Number of classes: {num_classes}=========================")

    test_dataset = ActionRecognitionDataset(test_seq, test_lbl)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_inference
    )

    # load T1 model
    unfreeze_layers = "entire"
    if unfreeze_layers is None:
        print("************Freezing all layers")
        t1 = load_T1("action_checkpoints/NUCLA_pretrained.pt", 
                    num_joints=NUM_JOINTS_NUCLA,
                    three_d=True,
                    d_model=hidden_size, 
                    nhead=n_heads, 
                    num_layers=num_layers, 
                    device=device
                )
    else:
        t1 = load_T1("action_checkpoints/NUCLA_finetuned_T1.pt",
                    num_joints=NUM_JOINTS_NUCLA,
                    three_d=True,
                    d_model=hidden_size, 
                    nhead=n_heads, 
                    num_layers=num_layers, 
                    device=device
                )
        print(f"************Unfreezing layers: {unfreeze_layers}")
    
    t2 = load_T2("action_checkpoints/NUCLA_finetuned_T2.pt", d_model=hidden_size, nhead=n_heads, num_layers=num_layers, device=device)
    # load the cross attention module
    cross_attn = load_cross_attn("action_checkpoints/NUCLA_finetuned_cross_attn.pt", d_model=hidden_size, device=device)

    # load the gait recognition head
    gait_head = GaitRecognitionHead(input_dim=hidden_size, num_classes=num_classes)
    gait_head.load_state_dict(torch.load("action_checkpoints/NUCLA_finetuned_head.pt", map_location="cpu"))
    gait_head = gait_head.to(device)

    print("Aha! All models loaded successfully!")
    print("=" * 100)

    # evaluate the model
    print("=" * 50)
    print("[INFO] Starting evaluation...")
    print("=" * 50)
    accuracy, _, _ = evaluate(
        test_loader,
        t1,
        t2,
        cross_attn,
        gait_head,
        device=device
    )

    print("=" * 50)
    print("[INFO] Evaluation completed!")
    print(f"Final Accuracy: {accuracy:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
