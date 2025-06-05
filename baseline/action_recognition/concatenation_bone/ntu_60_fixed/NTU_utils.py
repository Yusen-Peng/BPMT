import os
import numpy as np
import glob
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from base_dataset import ActionRecognitionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence

NUM_JOINTS_NTU = 25
OFFICIAL_XSUB_TRAIN_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
OFFICIAL_XVIEW_TRAIN_CAMERAS = [2, 3]

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

def collate_fn_finetuning(batch):
    batch, labels = zip(*batch)
    batch = pad_sequence(batch, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels, dim=0)
    return batch, labels

def read_ntu_skeleton_file(
    filepath: str,
    num_joints: int = NUM_JOINTS_NTU
) -> np.ndarray:
    """
    Read an NTU RGB+D .skeleton file and return a NumPy array
    of shape (T, 75) containing ONLY the first body's xyz coords.
    Pointer alignment is kept by skipping remaining bodies.
    """
    with open(filepath, "r") as f:
        total_frames = int(f.readline().strip())
        xyz = np.zeros((total_frames, num_joints, 3), dtype=np.float32)

        for t in range(total_frames):
            body_cnt = int(f.readline().strip())

            if body_cnt > 0:
                # skip over the metadata
                _ = f.readline().strip().split()
                _ = int(f.readline().strip())

                for j in range(num_joints):
                    vals = list(map(float, f.readline().strip().split()))
                    xyz[t, j] = vals[:3]
                
                # skip over the rest of bodies
                for _ in range(body_cnt - 1):
                    _ = f.readline().strip()
                    _ = int(f.readline().strip())
                    for _ in range(num_joints):
                        _ = f.readline().strip()

        # Crop to real frame count then flatten (T, 25*3)
        xyz = xyz[: min(total_frames, total_frames)]

        # reshape from (T, 25, 3) to (T, 75)
        return xyz.reshape(xyz.shape[0], -1)
    

def build_ntu_skeleton_lists_xsub(
    skeleton_root: str,
    is_train: bool = True,
    train_subjects: List[int] = OFFICIAL_XSUB_TRAIN_SUBJECTS,
    num_joints: int = NUM_JOINTS_NTU
) -> Tuple[List[np.ndarray], List[int]]:
    sequences, labels = [], []

    for filepath in tqdm(sorted(glob.glob(os.path.join(skeleton_root, '*.skeleton')))):
        filename = os.path.basename(filepath)
        subject_id = int(filename[9:12])

        if (is_train and subject_id not in train_subjects) or (not is_train and subject_id in train_subjects):
            continue

        action_idx = int(filename[17:20]) - 1
        skeleton = read_ntu_skeleton_file(filepath, num_joints)
        hip = skeleton[:, :3]
        skeleton = skeleton - np.tile(hip, (1, num_joints))
        sequences.append(skeleton)
        labels.append(action_idx)

    return sequences, labels

def build_ntu_skeleton_lists_xview(
    skeleton_root: str,
    is_train: bool = True,
    train_cameras: List[int] = OFFICIAL_XVIEW_TRAIN_CAMERAS,
    num_joints: int = NUM_JOINTS_NTU
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Build NTU RGB+D dataset using Cross-View split.
    Args:
        skeleton_root: path to folder with .skeleton files
        is_train: True for train split, False for test split
        train_cameras: list of camera IDs used for training (default: [2, 3])
        num_joints: number of joints (default: 25 for NTU)
    Returns:
        sequences, labels
    """
    sequences, labels = [], []

    for filepath in tqdm(sorted(glob.glob(os.path.join(skeleton_root, '*.skeleton')))):
        filename = os.path.basename(filepath)
        camera_id = int(filename[5:8])  # extract "C###" → camera ID

        # Check if sample belongs to current split
        if (is_train and camera_id not in train_cameras) or (not is_train and camera_id in train_cameras):
            continue

        # Extract action label (zero-based)
        action_idx = int(filename[17:20]) - 1

        skeleton = read_ntu_skeleton_file(filepath, num_joints)
        hip = skeleton[:, :3]
        skeleton = skeleton - np.tile(hip, (1, num_joints))

        sequences.append(skeleton)
        labels.append(action_idx)

    return sequences, labels


def split_train_val(
    sequences: List[np.ndarray],
    labels: List[int],
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    """
    Splits the NTU dataset into train and validation sets.
    """
    indices = np.arange(len(sequences))
    tr_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels
    )

    tr_seq  = [sequences[i] for i in tr_idx]
    tr_lbl  = [labels[i] for i in tr_idx]
    val_seq = [sequences[i] for i in val_idx]
    val_lbl = [labels[i] for i in val_idx]

    return tr_seq, tr_lbl, val_seq, val_lbl


if __name__ == "__main__":
    import time
    t_start = time.time()
    
    # all_seq, all_lbl = build_ntu_skeleton_lists_xsub('nturgb+d_skeletons', is_train=True)
    all_seq, all_lbl = build_ntu_skeleton_lists_xview('nturgb+d_skeletons', is_train=True)

    t_end = time.time()
    print(f"[INFO] Time taken to load NTU skeletons: {t_end - t_start:.2f} seconds")
    print(f"[VERIFY] Number of sequences: {len(all_seq)}")
    print(f"[VERIFY] Number of unique labels: {len(set(all_lbl))}")
    tr_seq, tr_lbl, val_seq, val_lbl = split_train_val(all_seq, all_lbl, val_ratio=0.15)
    train_set = ActionRecognitionDataset(tr_seq, tr_lbl)
    val_set = ActionRecognitionDataset(val_seq, val_lbl)
    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn_batch_padding
    )
    val_loader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn_batch_padding
    )