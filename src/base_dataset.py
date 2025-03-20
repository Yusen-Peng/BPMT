import os
import glob
import numpy as np
from typing import List, Tuple
import torch
from torch.utils.data import Dataset


class GaitRecognitionDataset(Dataset):
    """
        A dataset class for gait recognition encapsulating sequences of 2D keypoints and their corresponding labels.
        Args:
            sequences (List[np.ndarray]): A list of 2D keypoint sequences.
            labels (List[int]): A list of corresponding labels for each sequence.
        Attributes:
            sequences (List[np.ndarray]): The input sequences of 2D keypoints.
            labels (List[int]): The corresponding labels for each sequence.
            num_classes (int): The number of unique classes/labels.
        Methods:   
            __len__(): Returns the number of sequences in the dataset.
            __getitem__(idx): Returns the sequence and label at the given index.
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
    
        # convert to tensors
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return seq_tensor, label_tensor


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



# DEBUGGING
if __name__ == "__main__":
    root_dir = "2D_Poses_50/"
    batch_size = 4
    num_epochs = 100
    hidden_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 50)
    print(f"[INFO] Starting Gait3D dataset processing on {device}...")
    print("=" * 50)



    # Step 1: Load sequences and labels
    print(f"\n[INFO] Loading data from root directory: {root_dir}")
    sequences, labels = load_all_data(root_dir)
    print("\n[DEBUG] Data Statistics:")
    print(f"  - Total sequences loaded: {len(sequences)}")
    print(f"  - Total labels loaded: {len(labels)}")
    print(f"  - Unique label count: {len(set(labels))}")

    # step 2: verify sequence shapes
    print("\n[INFO] Verifying sequence shapes...")
    for i, seq in enumerate(sequences[:5]):  # Check first 5 sequences
        print(f"  - Sequence {i}: Shape {seq.shape}")

    # step 3: check label distribution
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print("\n[INFO] Label distribution:")
    for lbl, count in zip(unique_labels, label_counts):
        print(f"  - Label {lbl}: {count} sequences")

    # step 4: create dataset
    print("\n[INFO] Initializing dataset object...")
    dataset = GaitRecognitionDataset(sequences, labels)

    print("\n[DEBUG] Dataset Verification:")
    print(f"  - Dataset size: {len(dataset)}")
    print(f"  - Number of unique classes: {dataset.num_classes}")
    print("\n[INFO] Dataset setup complete. Ready for training!")
