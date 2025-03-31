import numpy as np
from typing import List
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

# # DEBUGGING
# if __name__ == "__main__":
#     root_dir = "2D_Poses_50/"
#     batch_size = 4
#     num_epochs = 100
#     hidden_size = 64
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print("=" * 50)
#     print(f"[INFO] Starting Gait3D dataset processing on {device}...")
#     print("=" * 50)

#     # Step 1: Load sequences and labels
#     print(f"\n[INFO] Loading data from root directory: {root_dir}")
#     sequences, labels = load_all_data(root_dir)
#     print("\n[DEBUG] Data Statistics:")
#     print(f"  - Total sequences loaded: {len(sequences)}")
#     print(f"  - Total labels loaded: {len(labels)}")
#     print(f"  - Unique label count: {len(set(labels))}")

#     # step 2: verify sequence shapes
#     print("\n[INFO] Verifying sequence shapes...")
#     for i, seq in enumerate(sequences[:5]):  # Check first 5 sequences
#         print(f"  - Sequence {i}: Shape {seq.shape}")

#     # step 3: check label distribution
#     unique_labels, label_counts = np.unique(labels, return_counts=True)
#     print("\n[INFO] Label distribution:")
#     for lbl, count in zip(unique_labels, label_counts):
#         print(f"  - Label {lbl}: {count} sequences")

#     # step 4: create dataset
#     print("\n[INFO] Initializing dataset object...")
#     dataset = GaitRecognitionDataset(sequences, labels)

#     print("\n[DEBUG] Dataset Verification:")
#     print(f"  - Dataset size: {len(dataset)}")
#     print(f"  - Number of unique classes: {dataset.num_classes}")
#     print("\n[INFO] Dataset setup complete. Ready for training!")
