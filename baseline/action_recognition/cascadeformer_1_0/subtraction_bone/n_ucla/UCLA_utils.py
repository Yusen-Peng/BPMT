import random
import numpy as np
import torch
from scipy.io import loadmat
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os
import glob
from tqdm import tqdm
from typing import List, Tuple, Dict

NUM_JOINTS_NUCLA = 20

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_nucla_skeleton_file(path: str) -> np.ndarray:
    """
    Loads the first person in a frame from N-UCLA by reading only the first 20 joints.
    Skips the first line (person count metadata).
    Returns: (20, 3) numpy array of joint coordinates.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 21:
        print(f"==[WARNING] {path[7:]} has insufficient data.==")
        return None

    joints = []
    for line in lines[1:21]:  # skip metadata, read 20 joints
        x, y, z, _ = map(float, line.strip().split(','))
        joints.append([x, y, z])

    # sanity check
    assert len(joints) == NUM_JOINTS_NUCLA, f"From {path} - Expected {NUM_JOINTS_NUCLA} joints, but got {len(joints)}"

    return np.array(joints, dtype=np.float32)

def build_nucla_action_lists_cross_view(
    root: str,
    train_views: List[str],
    test_views: List[str]
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:

    action_names = [f"a{str(i).zfill(2)}" for i in range(1, 13)]
    label2idx = {name: idx for idx, name in enumerate(action_names)}

    train_seq, train_lbl, test_seq, test_lbl = [], [], [], []
    all_views = train_views + test_views

    for view in all_views:
        view_path = os.path.join(root, view)
        action_dirs = sorted(glob.glob(os.path.join(view_path, 'a*_s*_e*')))

        for i, action_path in enumerate(action_dirs):
            action_name = os.path.basename(action_path).split('_')[0]
            label = label2idx[action_name]

            skeleton_files = sorted(glob.glob(os.path.join(action_path, '*_skeletons.txt')))


            seq = []
            for f in skeleton_files:
                # load the skeleton file
                skeleton = load_nucla_skeleton_file(f)
                if skeleton is not None:
                    seq.append(skeleton)

            # check if the sequence is empty
            if len(seq) == 0:
                print(f"======[WARNING] {action_path[7:]} from {view} has no valid skeletons. Skipping.======")
                continue

            # Stack all frames in the video
            seq = np.stack(seq, axis=0)  # shape (T, 20, 3)

            # Center each frame around the hip (joint 0)
            hip = seq[:, 0:1, :]  # shape (T, 1, 3)
            seq = seq - hip       # shape (T, 20, 3)

            # Flatten per frame: (T, 20, 3) → (T, 60)
            seq = seq.reshape(seq.shape[0], -1)  # (T, 60)

            # Normalize the *entire sequence*
            mean = seq.mean(axis=0, keepdims=True)  # shape (1, 60)
            std = seq.std(axis=0, keepdims=True) + 1e-8
            seq = (seq - mean) / std  # shape (T, 60)

            if view in train_views:
                train_seq.append(seq)
                train_lbl.append(label)
            else:
                test_seq.append(seq)
                test_lbl.append(label)

    print(f"#classes={len(label2idx)} | train_videos={len(train_seq)} | test_videos={len(test_seq)}")
    return train_seq, train_lbl, test_seq, test_lbl



def split_train_val(train_seq, train_lbl, val_ratio=0.15, seed=42):
    tr_idx, val_idx = train_test_split(
        np.arange(len(train_seq)),
        test_size=val_ratio,
        random_state=seed,
        stratify=train_lbl
    )
    tr_seq  = [train_seq[i] for i in tr_idx]
    tr_lbl  = [train_lbl[i] for i in tr_idx]
    val_seq = [train_seq[i] for i in val_idx]
    val_lbl = [train_lbl[i] for i in val_idx]

    return tr_seq, tr_lbl, val_seq, val_lbl


def collate_fn_batch_padding(batch):
    """
    a collate function for DataLoader that pads sequences to the maximum length in the batch.
    
    Returns:
      padded_seqs: (B, T_max, D) tensor
      labels: (B,) or (B, something)
      lengths: list of original sequence lengths
    """
    sequences, labels = zip(*batch)    
    padded_seq = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels, dim=0).long()
    return padded_seq, labels

def collate_fn_pairs(batch):
    """
    A collate function for second-stage pretraining.
    Pads two sets of variable-length sequences (modality A and modality B) separately.

    Args:
        batch: list of tuples [(xA1, xB1), (xA2, xB2), ...]
    
    Returns:
        xA_padded: (B, T_A_max, D_A)
        xB_padded: (B, T_B_max, D_B)
    """
    xA_list = [xA for xA, _ in batch]
    xB_list = [xB for _, xB in batch]

    xA_padded = pad_sequence(xA_list, batch_first=True, padding_value=0.0)
    xB_padded = pad_sequence(xB_list, batch_first=True, padding_value=0.0)

    return xA_padded, xB_padded


def collate_fn_finetuning(batch):
    batch, labels = zip(*batch)
    batch = pad_sequence(batch, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels, dim=0)
    return batch, labels


def collate_fn_inference(batch):
    batch, labels = zip(*batch)
    batch = pad_sequence(batch, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels, dim=0)
    return batch, labels

if __name__ == "__main__":
    set_seed(42)

    # Point to your dataset root
    root = "N_UCLA/"
    train_views = ['view_1', 'view_2']
    test_views  = ['view_3']

    # Run the loader
    train_seq, train_lbl, test_seq, test_lbl = build_nucla_action_lists_cross_view(
        root=root,
        train_views=train_views,
        test_views=test_views
    )

    # Check sample info
    print("\nSample shapes:")
    print(f"Train: {len(train_seq)} sequences | Shape[0]: {train_seq[0].shape}")
    print(f"Val:   {len(train_lbl)} labels | First label: {train_lbl[0]}")
    print(f"Test:  {len(test_seq)} sequences | Shape[0]: {test_seq[0].shape}")
    print(f"Test labels sample: {test_lbl[:-100]}")
