import os
import glob
import numpy as np
from typing import List, Tuple
import torch
from base_dataset import GaitRecognitionDataset
from base_dataset import load_all_data



class GaitRecognitionModalityAwareDataset(GaitRecognitionDataset):
    """
        A modality-aware dataset for gait recognition that treats different body regions as separate modalities.
    """

    # define keypoint modalities
    MODALITIES = {
        "torso": [0, 1, 2, 3, 4, 5, 6, 11, 12],  # Nose, Eyes, Ears, Shoulders, Hips
        "left_arm": [7, 9],  # Left Elbow, Left Wrist
        "right_arm": [8, 10],  # Right Elbow, Right Wrist
        "left_leg": [13, 15],  # Left Knee, Left Ankle
        "right_leg": [14, 16]  # Right Knee, Right Ankle
    }

    def __init__(self, sequences: List[np.ndarray], labels: List[int], modality: str):
        """
        Args:
            sequences (List[np.ndarray]): List of 2D keypoint sequences.
            labels (List[int]): Corresponding labels.
            modality (str): The body modality to extract ('torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg').
        """
        
        self.modality = modality
        self.indices = self.MODALITIES[modality]

        # extract only the selected modality keypoints
        sequences = [self.extract_modality(seq) for seq in sequences]

        # initialize the base class
        super().__init__(sequences, labels)
    
    
    def extract_modality(self, seq: np.ndarray) -> np.ndarray:
        """
            Extracts only the relevant keypoints for the selected modality.
            Args:
                seq (np.ndarray): Full 2D keypoint sequence (num_frames, num_joints * 2).
            Returns:
                np.ndarray: The extracted keypoints for the specified modality.
        """
        num_frames = seq.shape[0]
        selected_keypoints = np.zeros((num_frames, len(self.indices) * 2))
        
        for i, joint_idx in enumerate(self.indices):
            selected_keypoints[:, i * 2: (i + 1) * 2] = seq[:, joint_idx * 2: (joint_idx * 2) + 2]
        
        return selected_keypoints


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
    print("\n[INFO] Creating modality-aware datasets...")
    torso_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "torso")
    left_arm_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "left_arm")
    right_arm_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "right_arm")
    left_leg_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "left_leg")
    right_leg_modality = GaitRecognitionModalityAwareDataset(sequences, labels, "right_leg")

    # Step 5: confirm modality datasets
    print("\n[DEBUG] Modality Dataset Sizes:")
    print(f"  - Torso modality: {len(torso_modality)} samples")
    print(f"  - Left Arm modality: {len(left_arm_modality)} samples")
    print(f"  - Right Arm modality: {len(right_arm_modality)} samples")
    print(f"  - Left Leg modality: {len(left_leg_modality)} samples")
    print(f"  - Right Leg modality: {len(right_leg_modality)} samples")

    print("\n[INFO] Data partitioning complete! Each body region is now treated as a separate modality.")
    print("=" * 60)


    # Step 6: debug first 3 samples per modality
    print("\n[INFO] Checking first few samples from each modality...")
    for modality_name, modality_dataset in [
        ("Torso", torso_modality),
        ("Left Arm", left_arm_modality),
        ("Right Arm", right_arm_modality),
        ("Left Leg", left_leg_modality),
        ("Right Leg", right_leg_modality),
    ]:
        print(f"\n[DEBUG] Modality: {modality_name}")
        for i in range(min(3, len(modality_dataset))):
            seq_tensor, label_tensor = modality_dataset[i]
            print(f"  - Sample {i}:")
            print(f"    - Sequence tensor shape: {seq_tensor.shape} | Data type: {seq_tensor.dtype}")
            print(f"    - Label tensor: {label_tensor.item()}")
