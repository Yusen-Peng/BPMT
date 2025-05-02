import numpy as np
from typing import List
import torch
from torch.utils.data import Dataset

class ActionRecognitionDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], labels: List[int]):
        self.seqs   = sequences
        self.labels = labels
        self.num_classes = len(set(labels))

    def __len__(self): return len(self.seqs)

    def __getitem__(self, idx):
        seq   = torch.tensor(self.seqs[idx],   dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label
