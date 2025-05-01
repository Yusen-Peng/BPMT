import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import defaultdict
from utils import collate_fn_pairs
from tqdm import tqdm
from typing import Tuple, Dict
from pretraining import BaseT1
from torch.nn import functional as F
import matplotlib.pyplot as plt


def load_T1(model_path: str, num_joints: int = 17, d_model: int = 128, nhead: int = 4, num_layers: int = 2, freeze: bool = True,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> BaseT1:
    """
        loads a BaseT1 model from a checkpoint
    """

    model = BaseT1(num_joints=num_joints, d_model=d_model, nhead=nhead, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # optionally freeze the model parameters
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # move model to device and return the model
    return model.to(device)


class CrossAttention(nn.Module):
    """
    Simple cross-attention block that applies multi-head attention between two feature sequences.
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        # cross-attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # layer normalization
        self.norm = nn.LayerNorm(d_model)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:        
        # cross attention
        attn_out, _ = self.cross_attention(Q, K, V)

        # apply layer normalization and dropout, and residual connection
        out = self.norm(Q + self.dropout(attn_out))
        
        return out


class BaseT2(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super(BaseT2, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
    
    def encode(self, x):
        """
            Encodes the input sequence using the transformer encoder.
            Returns the encoded features.
        """
        encoded = self.encoder(x)
        return encoded


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """
    Implements Batch-Hard Triplet Loss.

    Args:
        embeddings (torch.Tensor): Embeddings of shape (batch_size, embed_dim)
        labels (torch.Tensor): Labels of shape (batch_size)
        margin (float): Margin for triplet loss

    Returns:
        torch.Tensor: Loss value (scalar)
    """

    # Compute pairwise distances
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    # mask positives and negatives
    labels = labels.unsqueeze(1)
    mask_positive = (labels == labels.T)
    mask_negative = (labels != labels.T)

    # for each anchor, find the hardest/furthest positive 
    dist_ap = (dist_matrix * mask_positive.float()).max(dim=1)[0]

    # for each anchor, find the hardest/closest negative
    dist_matrix_neg = dist_matrix + (1 - mask_negative.float()) * 1e6
    dist_an = dist_matrix_neg.min(dim=1)[0]

    # compute the triplet loss
    losses = F.relu(dist_ap - dist_an + margin)

    # mean over the batch
    return losses.mean()


from typing import List
def finetuning(
    train_loader: DataLoader,
    val_loader: DataLoader,
    t1: BaseT1,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    num_epochs: int = 200,
    lr: float = 1e-5,
    freezeT1: bool = True,
    unfreeze_layers: List[int] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[BaseT1, nn.Module]:
    
    # freeze T1 parameters
    if freezeT1:
        for param in t1.parameters():
            param.requires_grad = False

        # unfreeze specific layers if specified
        if unfreeze_layers is not None:
            for layer in unfreeze_layers:
                for param in t1.transformer_encoder.layers[layer].parameters():
                    param.requires_grad = True
    else:
        for param in t1.parameters():
            param.requires_grad = True

    t1.to(device)


    # intialize T2 transformer and cross-attention
    t2 = BaseT2(d_model, nhead, num_layers).to(device)
    cross_attn = CrossAttention(d_model, nhead).to(device)

    # optimizer and loss
    params = list(filter(lambda p: p.requires_grad, t1.parameters())) + \
         list(t2.parameters()) + \
         list(cross_attn.parameters())    
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)

    train_losses, val_losses = [], []

    for epoch in tqdm(range(num_epochs)):
        t2.train()
        cross_attn.train()

        t1_trainable = any(p.requires_grad for p in t1.parameters())
        t1.train(mode=t1_trainable)

        total_loss, correct, total = 0.0, 0, 0
        for i, (skeletons, labels) in enumerate(train_loader):
            skeletons, labels = skeletons.to(device), labels.to(device)

            if t1_trainable:
                x1 = t1.encode(skeletons)        # grads will flow
            else:
                with torch.no_grad():
                    x1 = t1.encode(skeletons)    # frozen model, no grads

            x2 = t2.encode(x1)
            fused = cross_attn(x1, x2, x2)

            # we need to do pooling
            pooled = fused.mean(dim=1)

            #print(f"[INFO] running batch hard triplet loss.. for batch {i+1}")
            loss = batch_hard_triplet_loss(pooled, labels, margin=0.3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            #correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        train_losses.append(avg_loss)
        

        # Validation
        t2.eval()
        cross_attn.eval()
        
        all_embeddings = []
        all_labels = []


        with torch.no_grad():
            for skeletons, labels in val_loader:
                skeletons, labels = skeletons.to(device), labels.to(device)
                x1 = t1.encode(skeletons)
                x2 = t2.encode(x1)
                fused = cross_attn(x1, x2, x2)

                # pooling
                pooled = fused.mean(dim=1)

                all_embeddings.append(pooled)
                all_labels.append(labels)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute pairwise distances
        distance_matrix = torch.cdist(all_embeddings, all_embeddings, p=2) 

        # for each sample, rank others by distance
        ranks = distance_matrix.argsort(dim=1)

        correct = 0
        total = all_labels.size(0)

        for i in range(total):
            # we need to skip the scenario where the sample is the same as itself
            for j in ranks[i]:
                if j == i:
                    continue
                
                if all_labels[j] == all_labels[i]:
                    # top-1 is correct
                    correct += 1
                    break 
                else:
                    # top-1 is not correct
                    correct += 0
                    break

        rank1_accuracy = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_loss:.4f}, Validation Rank-1 Accuracy = {rank1_accuracy:.4f}")

    # Plotting the training and validation losses
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.tight_layout()
    # save the figure
    plt.savefig("figures/retrieval_finetuning_loss.png")

    # return T2, and cross_attn
    return t2, cross_attn


def load_T2(model_path: str,d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> BaseT2:
    """
        loads a BaseT2 model from a checkpoint
    """
    model = BaseT2(d_model=d_model, nhead=nhead, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    for param in model.parameters():
        param.requires_grad = False
        
    # move model to device and return the model
    return model.to(device)

def load_cross_attn(path: str,
                    d_model: int = 128,
                    nhead: int = 4,
                    device: str = "cuda") -> CrossAttention:
    """
        loads a CrossAttention model from a checkpoint
    """
    layer = CrossAttention(d_model=d_model, nhead=nhead)
    layer.load_state_dict(torch.load(path, map_location="cpu"))
    return layer.to(device)