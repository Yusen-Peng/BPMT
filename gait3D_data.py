import os
import glob
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

def collect_subject_sequences(subject_folder: str) -> List[np.ndarray]:
    sequences = []
    for root, dirs, files in os.walk(subject_folder):
        if any(f.endswith('.txt') for f in files):
            seq_data = load_sequence(root)
            if seq_data is not None and seq_data.shape[0] > 0:
                sequences.append(seq_data)
    return sequences

def load_sequence(seq_folder: str, num_joints: int = 17) -> np.ndarray:
    txt_files = glob.glob(os.path.join(seq_folder, '*.txt'))
    if not txt_files:
        return None
    
    def get_frame_index(fp):
        fname = os.path.basename(fp)
        parts = fname.split('_f')
        if len(parts) < 2:
            return 0
        frame_str = parts[1].split('.')[0]
        return int(''.join(filter(str.isdigit, frame_str))) 
    
    txt_files = sorted(txt_files, key=get_frame_index)
    
    frames_2d = []
    for fp in txt_files:
        keypoints_2d = parse_pose_file(fp, num_joints)
        frames_2d.append(keypoints_2d.flatten())
    return np.vstack(frames_2d)

def parse_pose_file(file_path: str, num_joints: int = 17) -> np.ndarray:
    with open(file_path, 'r') as f:
        line = f.readline().strip()
    data = list(map(float, line.split(',')))

    keypoints_3d = np.array(data[2 : 2 + (num_joints * 3)]).reshape(num_joints, 3) 
    keypoints_2d = keypoints_3d[:, :2]
    return keypoints_2d

def load_all_data(root_dir: str) -> Tuple[List[np.ndarray], List[int]]:
    subject_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    subject_folders.sort()
    
    all_sequences = []
    all_labels = []

    # Create a mapping from original subject ID -> new contiguous label
    unique_subject_ids = sorted(set(subject_folders))  # Sort for consistent ordering
    subject_id_map = {int(subj): idx for idx, subj in enumerate(unique_subject_ids)}

    for subject_folder in subject_folders:
        full_path = os.path.join(root_dir, subject_folder)
        seqs = collect_subject_sequences(full_path)

        try:
            subject_id = int(subject_folder)  # Convert '0003' -> 3
        except ValueError:
            subject_id = subject_folders.index(subject_folder)  # Fallback if needed

        new_label = subject_id_map[subject_id]  # Convert to contiguous index

        for s in seqs:
            all_sequences.append(s)
            all_labels.append(new_label)  # Use the remapped label

    return all_sequences, all_labels

class GaitDataset(Dataset):
    """
    A PyTorch Dataset that holds sequences of 2D keypoints and subject labels.
    """
    def __init__(self, sequences: List[np.ndarray], labels: List[int]):
        self.sequences = sequences
        self.labels = labels
        self.num_classes = len(set(labels))
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return seq_tensor, label_tensor
