import os
import glob
import numpy as np
import torch
import argparse
from typing import List, Tuple
from itertools import combinations
from base_dataset import ActionRecognitionDataset
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F
from pretraining import train_T1, BaseT1
from finetuning import load_T1, finetuning, GaitRecognitionHead
#from first_phase_baseline import BaseT1, train_T1
#from second_phase_baseline import BaseT2, train_T2, load_T1
#from finetuning import GaitRecognitionHead, finetuning, load_T2, load_cross_attn

from penn_utils import set_seed, build_penn_action_lists, split_train_val, collate_fn_finetuning, NUM_JOINTS_PENN

def parse_args():
    parser = argparse.ArgumentParser(description="Gait Recognition Training")
    parser.add_argument("--pretrain", action='store_true', help="Run the stage of pretraining")
    parser.add_argument("--root_dir", type=str, default="Penn_Action/", help="Root directory of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for the model")
    parser.add_argument("--class_specific_split", action='store_true', help="Use class-specific split for training and validation")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cuda or cpu)")
    return parser.parse_args()


def main():
    set_seed(42)
    # masking_strategy = "frame", "global_joint"
    masking_strategy = "global_joint"
    mask_ratio = 0.3
    val_ratio = 0.05

    args = parse_args()
    root_dir = args.root_dir
    # get the number of classes from the root_dir by taking the trailing number
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size
    device = args.device
    pretrain = args.pretrain

    print(f"pretrain?: {pretrain}")

    # transformer parameters
    hidden_size = 256
    n_heads = 8
    num_layers = 4
    print(f"hidden_size: {hidden_size}")
    print(f"n_heads: {n_heads}")
    print(f"num_layers: {num_layers}")
    print(f"batch_size: {batch_size}")

    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 50)
    print(f"[INFO] Starting Penn Action dataset processing on {device}...")
    print("=" * 50)

    # load the dataset
    train_seq, train_lbl, _, _ = build_penn_action_lists(root_dir)
    train_seq, train_lbl, val_seq, val_lbl = split_train_val(train_seq, train_lbl, val_ratio=val_ratio)

    train_dataset = ActionRecognitionDataset(train_seq, train_lbl)
    val_dataset = ActionRecognitionDataset(val_seq, val_lbl)
        
    # get the number of classes
    num_classes = len(set(train_lbl))
    print(f"[INFO] Number of classes: {num_classes}")
    print("=" * 100)

    if pretrain == True: 
        """
            pretraining on the whole dataset
        """

        print(f"\n==========================")
        print(f"Starting Pretraining...")
        print(f"==========================")
        
        # instantiate the model
        three_d = False
        model = BaseT1(
            num_joints=NUM_JOINTS_PENN,
            three_d=three_d,
            d_model=hidden_size,
            nhead=n_heads,
            num_layers=num_layers,
        ).to(device)
        
        # training
        # dataset, model, num_epochs=50, batch_size=16, lr=1e-4, mask_ratio=0.15, device='cuda'):
        print(f"[INFO] Mask ratio: {mask_ratio * 100}%")
        print(f"[INFO] train/val split ratio: {val_ratio * 100}%")
        lr = 1e-4
        train_T1(
            masking_strategy=masking_strategy,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=model,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            mask_ratio=mask_ratio,
            device=device
        )

        print("[TEST] testing global joint masking" + "=" * 40)
        # save pretrained model
        torch.save(model.state_dict(), f"action_checkpoints/Penn_pretrained.pt")

        print("Aha! pretraining is done!")
        print("=" * 100)
    
    
    print("=" * 100)
    print("=" * 100)
    print("=" * 100)


    # load T1 models
    three_d = False
    t1 = load_T1(
        model_path="action_checkpoints/Penn_pretrained.pt",
        num_joints=NUM_JOINTS_PENN,
        three_d=three_d,
        d_model=hidden_size,
        nhead=n_heads,
        num_layers=num_layers,
        freeze=True,
        device=device
    )

    print("pretrained model loaded successfully!")

    train_finetuning_dataset = ActionRecognitionDataset(train_seq, train_lbl)
    val_finetuning_dataset = ActionRecognitionDataset(val_seq, val_lbl)


    train_finetuning_dataloader = torch.utils.data.DataLoader(
        train_finetuning_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_finetuning
    )

    val_finetuning_dataloader = torch.utils.data.DataLoader(
        val_finetuning_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_finetuning
    )

    gait_head_template = GaitRecognitionHead(input_dim=hidden_size, num_classes=num_classes).to(device)

    freezeT1 = False
    unfreeze_layers = None # freeze all layers

    if freezeT1 and (unfreeze_layers is None):
        print("[INFO] freezing the entire T1 model...")
    elif freezeT1 and (unfreeze_layers is not None):
        print(f"[INFO] layerwise finetuning...")
        print(f"[INFO] unfreezing layers: {unfreeze_layers}...")
    elif not freezeT1:
        print("[INFO] finetuning the entire T1 model...")


    # finetuning learning rate
    fn_lr = 3e-5


    trained_T2, train_cross_attn, train_head = finetuning(
        train_loader=train_finetuning_dataloader,
        val_loader=val_finetuning_dataloader,
        t1=t1,
        gait_head=gait_head_template,
        d_model=hidden_size,
        nhead=n_heads,
        num_layers=num_layers,
        num_epochs=num_epochs,
        lr=fn_lr,
        freezeT1=freezeT1,
        unfreeze_layers=unfreeze_layers,
        device=device
    )

    print("Aha! Finetuning completed successfully!")
    if unfreeze_layers is not None:
        print(f"[INFO] Unfreezing layers: {unfreeze_layers}...")

    # save the finetuned models
    torch.save(trained_T2.state_dict(), f"action_checkpoints/Penn_finetuned_T2.pt")
    torch.save(train_cross_attn.state_dict(), f"action_checkpoints/Penn_finetuned_cross_attn.pt")
    torch.save(train_head.state_dict(), f"action_checkpoints/Penn_finetuned_head.pt")

    if any(param.requires_grad for param in t1.parameters()):
        torch.save(t1.state_dict(), f"action_checkpoints/Penn_finetuned_T1.pt")

    print("Aha! finetuned models saved successfully!")


if __name__ == "__main__":
    main()