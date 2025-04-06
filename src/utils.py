import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import os
import glob
from typing import List, Tuple


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
    print(f"...Loading {len(txt_files)} files from {seq_folder}")
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


def collect_subject_sequences(subject_folder: str) -> List[np.ndarray]:
    """
        Collects all sequences of 2D keypoints from a given subject folder.
        Args:
            subject_folder (str): Path to the subject folder containing sequence files.
        Returns:
            List[np.ndarray]: A list of 2D keypoint sequences.
    """
    sequences = []
    for root, _, files in os.walk(subject_folder):
        if any(f.endswith('.txt') for f in files):
            seq_data = load_sequence(root)
            if seq_data is not None and seq_data.shape[0] > 0:
                sequences.append(seq_data)
    return sequences



def load_all_data(root_dir: str) -> Tuple[List[np.ndarray], List[int]]:
    """
        Loads all sequences of 2D keypoints and their corresponding labels from the given root directory.
        Args:
            root_dir (str): Path to the root directory containing subject folders.
        Returns:
            Tuple[List[np.ndarray], List[int]]: A tuple containing a list of 2D keypoint sequences and their corresponding labels.
    """    

    subject_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    subject_folders.sort()
    
    all_sequences = []
    all_labels = []

    # create a mapping from original subject ID -> new contiguous label
    unique_subject_ids = sorted(set(subject_folders))
    subject_id_map = {int(subj): idx for idx, subj in enumerate(unique_subject_ids)}

    for subject_folder in subject_folders:
        full_path = os.path.join(root_dir, subject_folder)
        seqs = collect_subject_sequences(full_path)

        try:
            # convert '0003' -> 3
            subject_id = int(subject_folder)  
        except ValueError:
            subject_id = subject_folders.index(subject_folder)

        # remap the subject ID to a contiguous label
        new_label = subject_id_map[subject_id]  

        for s in seqs:
            all_sequences.append(s)

            # use the new contiguous label
            all_labels.append(new_label)

    return all_sequences, all_labels


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
    """
    Collate function for finetuningDataset.
    Pads each modality across time dimension to the max length in the batch.
    Returns:
        - torso_batch:      [B, T_max_torso, D]
        - left_arm_batch:   [B, T_max_left_arm, D]
        - right_arm_batch:  [B, T_max_right_arm, D]
        - left_leg_batch:   [B, T_max_left_leg, D]
        - right_leg_batch:  [B, T_max_right_leg, D]
        - labels:           [B]
    """
    torso_batch, left_arm_batch, right_arm_batch, left_leg_batch, right_leg_batch, labels = zip(*batch)

    torso_batch = pad_sequence(torso_batch, batch_first=True, padding_value=0.0)
    left_arm_batch = pad_sequence(left_arm_batch, batch_first=True, padding_value=0.0)
    right_arm_batch = pad_sequence(right_arm_batch, batch_first=True, padding_value=0.0)
    left_leg_batch = pad_sequence(left_leg_batch, batch_first=True, padding_value=0.0)
    right_leg_batch = pad_sequence(right_leg_batch, batch_first=True, padding_value=0.0)

    labels = torch.stack(labels, dim=0)

    return torso_batch, left_arm_batch, right_arm_batch, left_leg_batch, right_leg_batch, labels


