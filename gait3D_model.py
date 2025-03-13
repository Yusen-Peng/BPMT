import os
import glob
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

class GaitCNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(GaitCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, lengths=None):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.squeeze(2)
        out = self.fc(x)      
        return out

def collate_function_cnn(batch):
    """
    Collate function for variable-length sequences for a CNN.
    We still pad them to (B, T_max, input_dim).
    """
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    lengths = [seq.shape[0] for seq in sequences]
    
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    labels_tensor = torch.stack(labels, dim=0)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return padded_sequences, labels_tensor, lengths_tensor

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        
        for batch in train_loader:
            sequences, labels, lengths = batch
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        train_acc = total_correct / total_samples
        train_loss = total_loss / total_samples
        
        
        
        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                sequences, labels, lengths = batch
                sequences = sequences.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)
                
                outputs = model(sequences, lengths)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_samples += labels.size(0)
        
        val_acc = val_correct / val_samples
        val_loss = val_loss / val_samples
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
