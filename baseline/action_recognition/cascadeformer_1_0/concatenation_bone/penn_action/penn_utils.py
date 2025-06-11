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
from base_dataset import ActionRecognitionDataset

NUM_JOINTS_PENN = 13

"""
    0  - head  
    1  - left_shoulder  
    2  - right_shoulder  
    3  - left_elbow  
    4  - right_elbow  
    5  - left_wrist  
    6  - right_wrist  
    7  - left_hip  
    8  - right_hip  
    9  - left_knee  
    10 - right_knee  
    11 - left_ankle  
    12 - right_ankle
"""

BONE_PAIRS = [
    (1, 0),  # left_shoulder - head
    (2, 0),  # right_shoulder - head
    (2, 1),  # right_shoulder - left_shoulder
    (3, 1),  # left_elbow - left_shoulder
    (4, 2),  # right_elbow - right_shoulder
    (5, 3),  # left_wrist - left_elbow
    (6, 4),  # right_wrist - right_elbow
    (7, 1),  # left_hip - left_shoulder
    (8, 2),  # right_hip - right_shoulder
    (9, 7),  # left_knee - left_hip
    (10, 8), # right_knee - right_hip
    (11, 9), # left_ankle - left_knee
    (12, 10),# right_ankle - right_knee
]

def joint2bone_penn(joints: torch.Tensor) -> torch.Tensor:
    """
    Converts 2D joint positions (T, 13, 2) to bone vectors (T, 26).

    Args:
        joints: (T, 13, 2) tensor of joint coordinates

    Returns:
        bone_vecs: (T, 26) = 13 bones x 2D (x, y)
    """

    bone_vecs = []
    for j1, j2 in BONE_PAIRS:
        joint_a = joints[:, j1, :]  # (T, 2)
        joint_b = joints[:, j2, :]  # (T, 2)
        concat_pair = torch.cat([joint_a, joint_b], dim=-1)  # (T, 4)
        bone_vecs.append(concat_pair)
    
    bone_tensor = torch.stack(bone_vecs, dim=1)  # (T, 13, 4)
    return bone_tensor.view(joints.shape[0], -1)  # (T, 52)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_mat_pose(mat_path: str, drop_occluded: bool = True):
    """
    Returns a (T, 13*2) array of [x,y] per frame.
    Missing joints -> 0 (or NaN if drop_occluded=False).
    """
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    x, y, vis = mat['x'], mat['y'], mat['visibility']
    if drop_occluded:
        x = np.where(vis, x, 0.)
        y = np.where(vis, y, 0.)
    poses = np.stack([x, y], axis=-1)          # (T, 13, 2)
    joints = torch.from_numpy(poses)
    bones = joint2bone_penn(joints).numpy()
    return bones.astype(np.float32), int(mat['train']), str(mat['action'])


def build_penn_action_lists(root: str
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    """
    Returns train_sequences, train_labels, test_sequences, test_labels.
    Label indices are 0 .. 15 in the order you encounter them.
    """
    label2idx: Dict[str,int] = {}
    train_seq, train_lbl, test_seq, test_lbl = [], [], [], []

    for mat_path in sorted(glob.glob(os.path.join(root, 'labels', '*.mat'))):
        seq, is_train, action = load_mat_pose(mat_path)

        # map action string -> int label
        if action not in label2idx:
            label2idx[action] = len(label2idx)
        lbl = label2idx[action]

        if is_train == 1:
            train_seq.append(seq)
            train_lbl.append(lbl)
        elif is_train == -1:
            test_seq.append(seq)
            test_lbl.append(lbl)
        else:
            raise ValueError(f"Invalid is_train value: {is_train}")

    print(f'#classes={len(label2idx)} | train videos={len(train_seq)} | test videos={len(test_seq)}')
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
    labels = torch.stack(labels, dim=0)
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
    root = "Penn_Action/"

    train_seq, train_lbl, test_seq, test_lbl = build_penn_action_lists(root)
    train_seq, train_lbl, val_seq, val_lbl = split_train_val(train_seq, train_lbl, val_ratio=0.15)

    train_ds = ActionRecognitionDataset(train_seq, train_lbl)
    val_ds   = ActionRecognitionDataset(val_seq, val_lbl)
    test_ds  = ActionRecognitionDataset(test_seq, test_lbl)
    
    # data loader
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn_batch_padding
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn_batch_padding
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn_batch_padding
    )

    # check the data loader
    print("Train dataset size:", len(train_loader.dataset))
    print("Example batch shape:", next(iter(train_loader))[0].shape)
    print("Validation dataset size:", len(val_loader.dataset))
    print("Example batch shape:", next(iter(val_loader))[0].shape)
    print("Test dataset size:", len(test_loader.dataset))
    print("Example batch shape:", next(iter(test_loader))[0].shape)
