import os
import glob
import numpy as np
import torch
import argparse
from typing import List, Tuple
from itertools import combinations
from base_dataset import GaitRecognitionDataset
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F
from pretraining import train_T1, BaseT1
from finetuning import load_T1, finetuning

from utils import set_seed, get_num_joints_for_modality, collate_fn_finetuning, aggregate_train_val_data_by_camera_split, collect_all_valid_subjects

def parse_args():
    parser = argparse.ArgumentParser(description="Gait Recognition Training")
    parser.add_argument("--pretrain", action='store_true', help="Run the stage of pretraining")
    parser.add_argument("--root_dir", type=str, default="2D_Poses_50/", help="Root directory of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for the model")
    parser.add_argument("--class_specific_split", action='store_true', help="Use class-specific split for training and validation")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cuda or cpu)")
    return parser.parse_args()


from collections import defaultdict
import random
from torch.utils.data import Sampler
class PKSampler(Sampler):

    def __init__(self, labels, P=8, K=4, shuffle=True):
        self.labels      = np.array(labels)
        self.label_index = defaultdict(list)
        for idx, lab in enumerate(labels):
            self.label_index[lab].append(idx)
        self.labels_set  = list(self.label_index.keys())
        self.P, self.K   = P, K
        self.shuffle     = shuffle
        self.num_batches = len(labels) // (P * K)

    def __iter__(self):
        labels = self.labels_set.copy()
        if self.shuffle:
            random.shuffle(labels)

        for _ in range(self.num_batches):
            chosen_labels = random.sample(labels, self.P)
            batch = []
            for lab in chosen_labels:
                idxs = self.label_index[lab]
                if len(idxs) >= self.K:
                    batch.extend(random.sample(idxs, self.K))
                else:
                    batch.extend(random.choices(idxs, k=self.K))
            yield batch

    def __len__(self):
        return self.num_batches


def main():
    set_seed(42)

    args = parse_args()
    root_dir = args.root_dir
    # get the number of classes from the root_dir by taking the trailing number
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size
    device = args.device
    pretrain = args.pretrain

    print(f"pretrain?: {pretrain}")

    # Set the device

    hidden_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 50)
    print(f"[INFO] Starting Gait3D dataset processing on {device}...")
    print("=" * 50)

    MIN_CAMERAS = 1 # by setting this to 1, we are using the entire dataset here

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

    # dataset creation of the entire skeleton
    train_dataset = GaitRecognitionDataset(train_sequences, train_labels)
    val_dataset = GaitRecognitionDataset(val_sequences, val_labels)

    # figure out how many joints
    num_joints = get_num_joints_for_modality("Torso") + get_num_joints_for_modality("Left_Arm") + \
        get_num_joints_for_modality("Right_Arm") + get_num_joints_for_modality("Left_Leg") + \
        get_num_joints_for_modality("Right_Leg")


    if pretrain == True: 
        """
            pretraining on the whole dataset
        """

        print(f"\n==========================")
        print(f"Starting Pretraining...")
        print(f"==========================")
        
        
        print(f"[INFO] Number of joints in skeletons: {num_joints}")

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
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=model,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=1e-5,
            mask_ratio=0.15,
            device=device
        )

        # save pretrained model
        torch.save(model.state_dict(), f"baseline_checkpoints/pretrained.pt")

        print("Aha! pretraining is done!")
        print("=" * 100)
    
    
    print("=" * 100)
    print("=" * 100)
    print("=" * 100)


    # load T1 models
    t1 = load_T1(
        model_path="baseline_checkpoints/pretrained.pt",
        num_joints=num_joints,
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        freeze=True,
        device=device
    )

    print("pretrained model loaded successfully!")
   
    train_finetuning_dataset = GaitRecognitionDataset(train_sequences, train_labels)
    val_finetuning_dataset = GaitRecognitionDataset(val_sequences, val_labels)


    # do PK sampling for training
    P = 4 # number of classes per batch
    K = 4 # number of samples per class
    batch_sampler = PKSampler(train_labels, P=P, K=K)
    train_finetuning_dataloader = torch.utils.data.DataLoader(
        train_finetuning_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn_finetuning
    )


    val_finetuning_dataloader = torch.utils.data.DataLoader(
        val_finetuning_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_finetuning
    )

    freezeT1 = False
    unfreeze_layers = None
    if unfreeze_layers is not None:
        print(f"[INFO] Unfreezing layers: {unfreeze_layers}...")


    trained_T2, train_cross_attn = finetuning(
        train_loader=train_finetuning_dataloader,
        val_loader=val_finetuning_dataloader,
        t1=t1,
        d_model=hidden_size,
        nhead=4,
        num_layers=2,
        num_epochs=num_epochs,
        lr=1e-5,
        freezeT1=freezeT1,
        unfreeze_layers=unfreeze_layers,
        device=device
    )

    print("Aha! Finetuning completed successfully!")
    

    # save the finetuned models
    torch.save(trained_T2.state_dict(), f"baseline_checkpoints/finetuned_T2.pt")
    torch.save(train_cross_attn.state_dict(), f"baseline_checkpoints/finetuned_cross_attn.pt")

    
    if any(param.requires_grad for param in t1.parameters()):
        torch.save(t1.state_dict(), f"baseline_checkpoints/finetuned_T1.pt")

    print("Aha! finetuned models saved successfully!")


if __name__ == "__main__":
    main()