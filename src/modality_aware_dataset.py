import os
import glob
import numpy as np
from typing import List, Tuple
import torch
from base_dataset import GaitRecognitionDataset
from utils import load_all_data

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



class PairwiseModalityDataset(torch.utils.data.Dataset):
    """
    Combines two aligned single-modality datasets into a dataset that returns (xA, xB) pairs.
    Assumes that both datasets are the same length and order.
    """
    def __init__(self, datasetA, datasetB):
        assert len(datasetA) == len(datasetB), "Datasets must be the same length"
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __len__(self):
        return len(self.datasetA)

    def __getitem__(self, idx):
        xA, _ = self.datasetA[idx]
        xB, _ = self.datasetB[idx]
        return xA, xB
