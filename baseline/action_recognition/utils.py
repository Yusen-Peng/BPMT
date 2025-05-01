import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os
import glob
from tqdm import tqdm
from typing import List, Tuple, Dict

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_pose_file(file_path: str, num_joints: int = 17) -> np.ndarray:
    """
        Parses a single pose file to extract 2D keypoints.
        Args:
            file_path (str): Path to the pose file.
            num_joint (int): Number of joints/keypoints per frame.
        Returns:
            np.ndarray: A 2D array of shape (num_joints, 2) containing the x and y coordinates of the keypoints.
    """
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        line = f.readline().strip()
    data = list(map(float, line.split(',')))

    keypoints_2d_with_confidence = np.array(data[2 : 2 + (num_joints * 3)]).reshape(num_joints, 3)
    
    # extract only the x and y coordinates without the confidence score
    keypoints_2d = keypoints_2d_with_confidence[:, :2]
    return keypoints_2d


def load_sequence(seq_folder: str, num_joints: int = 17) -> np.ndarray:
    """
        Loads a sequence of 2D keypoints from text files in the given folder.
        Args:
            seq_folder (str): Path to the folder containing the sequence files.
            num_joints (int): Number of joints/keypoints per frame.
        Returns:
            np.ndarray: A 2D array of shape (num_frames, num_joints * 2) containing the 2D keypoints.
    """
    txt_files = glob.glob(os.path.join(seq_folder, '*.txt'))
    if not txt_files:
        return None
    
    def get_frame_index(fp):
        """
            Extracts the frame index from the filename.
            Args:
                fp (str): File path.
            Returns:
                int: Frame index extracted from the filename.
        """
        fname = os.path.basename(fp)
        parts = fname.split('_f')
        if len(parts) < 2:
            return 0
        frame_str = parts[1].split('.')[0]
        return int(''.join(filter(str.isdigit, frame_str))) 
    
    # sort files based on frame index
    txt_files = sorted(txt_files, key=get_frame_index)

    # load keypoints from each file
    # shape of each keypoint: (num_joints, 3) -> (x, y, confidence)
    frames_2d = []
    for fp in txt_files:
        keypoints_2d = parse_pose_file(fp, num_joints)
        frames_2d.append(keypoints_2d.flatten())
    return np.vstack(frames_2d)

def collect_all_valid_subjects(parent_folder: str, min_cameras: int = 5) -> Dict[str, Dict[str, List[Tuple[np.ndarray, int]]]]:


    # subject ID to label mapping
    subject_ids = sorted([
    d for d in os.listdir(parent_folder)
    if os.path.isdir(os.path.join(parent_folder, d)) and not d.startswith(".")
    ])
    subject_id_to_label = {sid: idx for idx, sid in enumerate(subject_ids)}

    # subject id: data
    valid_subjects = {}

    # iterate over each subject
    for subject_id in sorted(os.listdir(parent_folder)):        
        subject_path = os.path.join(parent_folder, subject_id)
        if not os.path.isdir(subject_path):
            continue
        
        # map subject_id like 0013 to a number
        label = subject_id_to_label[subject_id]

        # collect sequences + labels for each camera
        cam_seqs_labels = collect_sequences_by_camera(subject_path, label=label)

        if len(cam_seqs_labels) >= min_cameras:
            valid_subjects[subject_id] = cam_seqs_labels

    # print the number of valid subjects
    print(f"Number of valid subjects with at least {min_cameras} cameras: {len(valid_subjects)}")
    
    return valid_subjects


def collect_sequences_by_camera(subject_folder: str, label: int) -> Dict[str, List[Tuple[np.ndarray, int]]]:
    sequences_with_labels_by_cam: Dict[str, List[Tuple[np.ndarray, int]]] = {}

    for cam_folder in os.listdir(subject_folder):
        cam_path = os.path.join(subject_folder, cam_folder)

        if not os.path.isdir(cam_path):
            continue

        if "_" in cam_folder and cam_folder.startswith("camid"):
            cam_id = cam_folder.split("_")[0]

            for seq_folder in os.listdir(cam_path):
                seq_path = os.path.join(cam_path, seq_folder)
                if not os.path.isdir(seq_path):
                    continue

                if any(f.endswith('.txt') for f in os.listdir(seq_path)):
                    seq_data = load_sequence(seq_path)
                    if seq_data is not None and seq_data.shape[0] > 0:
                        if cam_id not in sequences_with_labels_by_cam:
                            sequences_with_labels_by_cam[cam_id] = []
                        sequences_with_labels_by_cam[cam_id].append((seq_data, label))

    return sequences_with_labels_by_cam

def aggregate_train_val_data_by_camera_split(
    valid_subjects: Dict[str, Dict[str, List[Tuple[np.ndarray, int]]]],
    train_ratio: float = 0.75,
    seed: int = 42
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:

    random.seed(seed)
    train_sequences = []
    train_labels = []
    val_sequences = []
    val_labels = []

    for subject_id, cam_seqs_with_labels in valid_subjects.items():
        cam_ids = sorted(cam_seqs_with_labels.keys())
        num_total = len(cam_ids)

        # determine number of cameras for training
        num_train = max(1, int(num_total * train_ratio))
        # also ensure that we have at least one camera for validation
        num_train = min(num_train, num_total - 1)

        random.shuffle(cam_ids)
        train_cams = cam_ids[:num_train]
        val_cams = cam_ids[num_train:]

        # Aggregate sequences
        for cam in train_cams:
            for seq_data, label in cam_seqs_with_labels[cam]:
                train_sequences.append(seq_data)
                train_labels.append(label)
        for cam in val_cams:
            for seq_data, label in cam_seqs_with_labels[cam]:
                val_sequences.append(seq_data)
                val_labels.append(label)
    
    return train_sequences, train_labels, val_sequences, val_labels


def get_num_joints_for_modality(modality_name):
    """
        Returns the number of joints for a given modality.
        Args:
            modality_name (str): Name of the body modality.
        Returns:
            int: Number of joints for the specified modality.
    """
    if modality_name == "Torso":
        return 9
    elif modality_name in ["Left_Arm", "Right_Arm", "Left_Leg", "Right_Leg"]:
        return 2
    else:
        raise ValueError("Unknown modality")


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
    # Example usage
    root_dir = "2D_Poses_50/"
    MIN_CAMERAS = 3
    valid_subjects = collect_all_valid_subjects(root_dir, min_cameras=3)

    print(f"the number of valid subjects: {len(valid_subjects)}")

    train_sequences, train_labels, val_sequences, val_labels = aggregate_train_val_data_by_camera_split(
        valid_subjects,
        train_ratio=0.75,
        seed=42
    )