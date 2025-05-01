import numpy as np
from typing import List
from collections import defaultdict
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

        # Build label -> list of indices map
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
    
        # convert to tensors
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return seq_tensor, label_tensor
