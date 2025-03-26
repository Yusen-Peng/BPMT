import os
import glob
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from gait3D_data import load_all_data, GaitDataset
from gait3D_model import GaitCNN, collate_function_cnn, train_model

def main():
    """
        data preparation.
    """
    root_dir = "Gait3D-Benchmark/datasets/Gait3D/2D_Poses_mini/"
    batch_size = 4
    num_epochs = 100
    hidden_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sequences, labels = load_all_data(root_dir)
    dataset = GaitDataset(sequences, labels)
    num_classes = dataset.num_classes
    # 80/20 training-validation split
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function_cnn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_function_cnn)

    """
        train the model.
    """
    sample_seq, _ = dataset[0]
    input_dim = sample_seq.shape[1]
    model = GaitCNN(input_dim, num_classes=num_classes)
    
    # Train
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=1e-3, device=device)
    
    # Save the model
    torch.save(model.state_dict(), "gait_cnn_model.pth")
    print("CNN training complete and saved to gait_cnn_model.pth")

if __name__ == "__main__":
    main()
