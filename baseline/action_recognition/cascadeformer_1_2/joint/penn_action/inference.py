import torch
from sklearn.metrics import accuracy_score
import argparse
from typing import Tuple
from torch import nn
from torch.utils.data import DataLoader
from base_dataset import ActionRecognitionDataset
from penn_utils import set_seed, build_penn_action_lists, split_train_val, collate_fn_inference
from finetuning import load_T1, load_T2, load_cross_attn, GaitRecognitionHead

def count_all_parameters(
        T1: nn.Module, 
        T2: nn.Module, 
        cross_attn: nn.Module, 
        gait_head: nn.Module
    ) -> int:
    """
    Counts the total number of parameters in the T1, T2, cross-attention, and gait head models.
    
    Args:
        T1: T1 transformer model
        T2: T2 transformer model
        cross_attn: CrossAttention module
        gait_head: GaitRecognitionHead module

    Returns:
        total_params: Total number of parameters across all models
    """
    total_params = sum(p.numel() for p in T1.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in T2.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in cross_attn.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in gait_head.parameters() if p.requires_grad)
    
    return total_params

def evaluate(
    data_loader: DataLoader,
    t1: nn.Module,
    t2: nn.Module,
    cross_attn: nn.Module,
    gait_head: nn.Module,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Performs inference and computes accuracy over the given dataset.

    Args:
        data_loader: DataLoader for evaluation
        t1: pretrained (frozen or finetuned) T1 transformer
        t2: trained T2 transformer
        cross_attn: trained CrossAttention module
        gait_head: trained classification head
        device: device to run inference on
        pooling: pooling strategy - 'mean' or 'attention'
        attention_pool: optional attention pooling module (required if pooling == 'attention')

    Returns:
        accuracy: float
        all_preds: tensor of predictions
        all_labels: tensor of ground-truth labels
    """
    t1.eval()
    t2.eval()
    cross_attn.eval()
    gait_head.eval()
   

    all_preds, all_labels = [], []

    with torch.no_grad():
        for skeletons, labels in data_loader:
            skeletons, labels = skeletons.to(device), labels.to(device)

            x1 = t1.encode(skeletons)
            x2 = t2.encode(x1)
            fused = cross_attn(x1, x2, x2)
            pooled = fused.mean(dim=1)

            logits = gait_head(pooled)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)

    return accuracy, all_preds, all_labels

def parse_args():
    parser = argparse.ArgumentParser(description="Gait Recognition Inference")
    parser.add_argument("--root_dir", type=str, default="Penn_Action/", help="Root directory of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for Inference")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for the model")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cuda or cpu)")
    return parser.parse_args()


def main():
    set_seed(42)

    args = parse_args()
    root_dir = args.root_dir
    # get the number of classes from the root_dir by taking the trailing number
    batch_size = args.batch_size
    device = args.device

    # Set the device

    hidden_size = 256
    n_heads = 8
    num_layers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 50)
    print(f"[INFO] Starting Penn Action dataset processing on {device}...")
    print("=" * 50)

    # load the dataset
    train_seq, train_lbl, test_seq, test_lbl = build_penn_action_lists(root_dir)
    train_seq, train_lbl, val_seq, val_lbl = split_train_val(train_seq, train_lbl, val_ratio=0.05)
    
    test_dataset = ActionRecognitionDataset(test_seq, test_lbl)
    
    # get the number of classes
    num_classes = len(set(test_lbl))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_inference
    )

    # load T1 model
    unfreeze_layers = "entire"
    if unfreeze_layers is None:
        print("************Freezing all layers")
        t1 = load_T1("action_checkpoints/Penn_pretrained.pt", d_model=hidden_size, nhead=n_heads, num_layers=num_layers, device=device)
    else:
        t1 = load_T1("action_checkpoints/Penn_finetuned_T1.pt", d_model=hidden_size, nhead=n_heads, num_layers=num_layers, device=device)
        print(f"************Unfreezing layers: {unfreeze_layers}")
    
    t2 = load_T2("action_checkpoints/Penn_finetuned_T2.pt", d_model=hidden_size, nhead=n_heads, num_layers=num_layers, device=device)
    # load the cross attention module
    cross_attn = load_cross_attn("action_checkpoints/Penn_finetuned_cross_attn.pt", d_model=hidden_size, device=device)

    # load the gait recognition head
    gait_head = GaitRecognitionHead(input_dim=hidden_size, num_classes=num_classes)
    gait_head.load_state_dict(torch.load("action_checkpoints/Penn_finetuned_head.pt", map_location="cpu"))
    gait_head = gait_head.to(device)

    print("Aha! All models loaded successfully!")
    print("=" * 100)
    print("the total number of parameters in the model is: ")
    total_params = count_all_parameters(t1, t2, cross_attn, gait_head)
    print(f"Total parameters: {total_params:,}")
    print("=" * 100)

    # evaluate the model
    print("=" * 50)
    print("[INFO] Starting evaluation...")
    print("=" * 50)
    accuracy, _, _ = evaluate(
        test_loader,
        t1,
        t2,
        cross_attn,
        gait_head,
        device=device
    )

    print("=" * 50)
    print("[INFO] Evaluation completed!")
    print(f"Final Accuracy: {accuracy:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
