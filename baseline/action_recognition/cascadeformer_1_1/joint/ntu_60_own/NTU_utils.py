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

def normalize_scale(skeleton: np.ndarray) -> np.ndarray:
    """
    Normalize the skeleton so that the scale is invariant.
    E.g., divide by std or max joint distance to origin.
    """
    scale = np.linalg.norm(skeleton, axis=-1).max() + 1e-8
    return skeleton / scale

def add_gaussian_noise(sequence: np.ndarray, std=0.01) -> np.ndarray:
    noise = np.random.normal(0, std, sequence.shape)
    return sequence + noise

def random_rotation(sequence: np.ndarray, max_angle=10) -> np.ndarray:
    angle = np.deg2rad(np.random.uniform(-max_angle, max_angle))
    cos, sin = np.cos(angle), np.sin(angle)
    R = np.array([
        [cos, -sin, 0],
        [sin,  cos, 0],
        [0,     0,  1]
    ])
    return np.einsum('tjd,dk->tjk', sequence, R)

def random_scaling(sequence: np.ndarray, scale_range=(0.9, 1.1)) -> np.ndarray:
    scale = np.random.uniform(*scale_range)
    return sequence * scale

def random_frame_dropout(sequence: np.ndarray, drop_prob=0.1) -> np.ndarray:
    T = sequence.shape[0]
    mask = np.random.rand(T) > drop_prob
    return sequence[mask]

def temporal_crop(sequence: np.ndarray, crop_len=50) -> np.ndarray:
    T = sequence.shape[0]
    if T <= crop_len:
        return sequence
    start = np.random.randint(0, T - crop_len)
    return sequence[start:start + crop_len]

def temporal_jitter(sequence: np.ndarray, max_jitter=5) -> np.ndarray:
    T = sequence.shape[0]
    jitter = np.random.randint(-max_jitter, max_jitter + 1, size=T)
    jitter = np.clip(np.arange(T) + jitter, 0, T - 1)
    return sequence[jitter]


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
        #return xyz.reshape(xyz.shape[0], -1)

        # FIXME: no reshape!
        return xyz

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
        hip = skeleton[:, 0:1, :]  # shape: (T, 1, 3)
        skeleton = skeleton - hip  # shape: (T, 25, 3) - (T, 1, 3) → (T, 25, 3)

        # ADD NORMALIZATION
        skeleton = normalize_scale(skeleton)

        # HAHA! LET'S DO SOME AUGMENTATIONS
        if is_train:
            threshold = 0.2 # FIXME: WE CAN TUNE THIS
            if np.random.rand() < threshold:
                skeleton = add_gaussian_noise(skeleton, std=0.01)
            if np.random.rand() < threshold:
                skeleton = random_rotation(skeleton, max_angle=10)
            if np.random.rand() < threshold:
                skeleton = random_scaling(skeleton, scale_range=(0.9, 1.1))
            # if np.random.rand() < threshold:
            #     skeleton = random_frame_dropout(skeleton, drop_prob=0.1)
            if np.random.rand() < threshold:
                skeleton = temporal_crop(skeleton, crop_len=50)
            if np.random.rand() < threshold:
                skeleton = temporal_jitter(skeleton, max_jitter=5)

        sequences.append(skeleton)
        labels.append(action_idx)

    # cache them
    save_cached_data(sequences, labels, path="ntu_cache_train_sub.npz")

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

        skeleton = read_ntu_skeleton_file(filepath, num_joints) # (T, 25, 3)
        hip = skeleton[:, 0:1, :]  # shape: (T, 1, 3)
        skeleton = skeleton - hip  # shape: (T, 25, 3) - (T, 1, 3) → (T, 25, 3)

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

def save_cached_data(sequences, labels, path="ntu_cache_train_sub.npz"):
    sequences_obj = np.array(sequences, dtype=object)
    labels_arr = np.array(labels)
    np.savez_compressed(path, sequences=sequences_obj, labels=labels_arr)


if __name__ == "__main__":
    import time
    t_start = time.time()
    
    all_seq, all_lbl = build_ntu_skeleton_lists_xsub('nturgb+d_skeletons', is_train=True)
    #all_seq, all_lbl = build_ntu_skeleton_lists_xview('nturgb+d_skeletons', is_train=True)

    t_end = time.time()
    print(f"[INFO] Time taken to load NTU skeletons: {t_end - t_start:.2f} seconds")
    print(f"[VERIFY] Number of sequences: {len(all_seq)}")
    print(f"[VERIFY] Number of unique labels: {len(set(all_lbl))}")
    tr_seq, tr_lbl, val_seq, val_lbl = split_train_val(all_seq, all_lbl, val_ratio=0.15)

    # check the shape of a sample sequence
    print(f"[VERIFY] Sample sequence shape: {tr_seq[0].shape}")
    print(f"[VERIFY] Sample sequence shape: {tr_seq[15].shape}")

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