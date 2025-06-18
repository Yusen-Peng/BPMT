import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from penn_utils import collate_fn_pairs
import copy
from tqdm import tqdm
from typing import Tuple, Dict, Optional
from pretraining import BaseT1
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score

def evaluate(
    data_loader: DataLoader,
    t1: nn.Module,
    t2: nn.Module,
    cross_attn: nn.Module,
    gait_head: nn.Module,
    device: str = 'cuda',
) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Unified evaluation function matching training-time validation.

    Returns:
        - accuracy (float)
        - average loss (float, 0.0 if no criterion)
        - all_preds (optional)
        - all_labels (optional)
    """
    t1.eval()
    t2.eval()
    cross_attn.eval()
    gait_head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for skeletons, labels in data_loader:
            skeletons, labels = skeletons.to(device), labels.to(device)

            x1 = t1.encode(skeletons)
            x2 = t2.encode(x1)
            fused = cross_attn(x1, x2, x2)
            pooled = fused.mean(dim=1)

            logits = gait_head(pooled)


            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    
    return acc


def load_T1(model_path: str, num_joints: int = 13, three_d: bool = False, d_model: int = 128, nhead: int = 4, num_layers: int = 2, freeze: bool = True,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> BaseT1:
    """
        loads a BaseT1 model from a checkpoint
    """

    model = BaseT1(num_joints=num_joints, three_d=three_d, d_model=d_model, nhead=nhead, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # optionally freeze the model parameters
    if freeze:
        model.eval()
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


class GaitRecognitionHead(nn.Module):
    """
        A simple linear head for gait recognition.
        The model consists of a linear layer that maps the output of the transformer to the number of classes.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

from typing import List
def finetuning(
    train_loader: DataLoader,
    val_loader: DataLoader,
    t1: BaseT1,
    gait_head: nn.Module,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    num_epochs: int = 200,
    lr: float = 1e-5,
    wd: float = 1e-2,
    freezeT1: bool = True,
    unfreeze_layers: List[int] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[BaseT1, nn.Module, nn.Module]:
    
    print(f"is T1 freezed? {freezeT1}")
    print(f"unfreezing layers: {unfreeze_layers}")

    
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
    gait_head.to(device)

    # intialize T2 transformer and cross-attention
    t2 = BaseT2(d_model, nhead, num_layers).to(device)
    cross_attn = CrossAttention(d_model, nhead).to(device)

    # optimizer and loss
    params = list(filter(lambda p: p.requires_grad, t1.parameters())) + \
         list(t2.parameters()) + \
         list(cross_attn.parameters()) + \
         list(gait_head.parameters())

    #optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)
    # weight decay = 1e-4, 5e-5
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    num_training_steps = num_epochs
    num_warmup_steps = int(0.05 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_state = {
        't1': None,
        't2': None,
        'cross_attn': None,
        'gait_head': None
    }

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in tqdm(range(num_epochs)):
        gait_head.train()
        t2.train()
        cross_attn.train()

        t1_trainable = any(p.requires_grad for p in t1.parameters())
        t1.train(mode=t1_trainable)


        total_loss, correct, total = 0.0, 0, 0
        for skeletons, labels in train_loader:
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
            logits = gait_head(pooled)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = total_loss / total
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        
        # learning rate scheduler step
        scheduler.step()

        val_acc = evaluate(
            val_loader,
            t1,
            t2,
            cross_attn,
            gait_head,
            device=device
        )
        print(f"Epoch {epoch+1}/{num_epochs}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # NOTE: USE DEEP COPY INSTEAD OF SHALLOW COPY!!
            best_state['t1'] = copy.deepcopy(t1.state_dict())
            best_state['t2'] = copy.deepcopy(t2.state_dict())
            best_state['cross_attn'] = copy.deepcopy(cross_attn.state_dict())
            best_state['gait_head'] = copy.deepcopy(gait_head.state_dict())
            print(f"✅ Best model updated at epoch {epoch+1} with val acc: {val_acc:.4f}")

    t1.load_state_dict(best_state['t1'])
    t2.load_state_dict(best_state['t2'])
    cross_attn.load_state_dict(best_state['cross_attn'])
    gait_head.load_state_dict(best_state['gait_head'])
    return t1, t2, cross_attn, gait_head


def load_T2(model_path: str,d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> BaseT2:
    """
        loads a BaseT2 model from a checkpoint
    """
    model = BaseT2(d_model=d_model, nhead=nhead, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model.eval()
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
    layer.eval()
    for param in layer.parameters():
        param.requires_grad = False
    return layer.to(device)