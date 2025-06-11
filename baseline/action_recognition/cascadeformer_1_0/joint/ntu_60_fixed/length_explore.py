import os
import glob
import numpy as np
import torch
import argparse
from typing import List, Tuple
from itertools import combinations
from base_dataset import ActionRecognitionDataset
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from NTU_pretraining import train_T1, BaseT1
from finetuning import load_T1, finetuning, GaitRecognitionHead
#from first_phase_baseline import BaseT1, train_T1
#from second_phase_baseline import BaseT2, train_T2, load_T1
#from finetuning import GaitRecognitionHead, finetuning, load_T2, load_cross_attn

from penn_utils import set_seed
from NTU_utils import build_ntu_skeleton_lists_xsub, split_train_val, NUM_JOINTS_NTU, collate_fn_finetuning

# TODO: tune this
MAX_SEQ_LEN = 10


def preprocess_sequence(seq: np.ndarray, max_len: int = MAX_SEQ_LEN):
    T, D = seq.shape
    if T >= max_len:
        return seq[:max_len]
    else:
        pad = np.zeros((max_len - T, D), dtype=np.float32)
        return np.concatenate([seq, pad], axis=0)

def collate_fn_stack(batch):
    sequences, labels = zip(*batch)
    return torch.stack([torch.tensor(x) for x in sequences]), torch.tensor(labels)


def main():
    set_seed(42)
    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 50)
    print(f"[INFO] Starting NTU dataset processing on {device}...")
    print("=" * 50)

    # load the dataset
    import time
    t_start = time.time()
    all_seq, all_lbl = build_ntu_skeleton_lists_xsub('nturgb+d_skeletons', is_train=True)
    import matplotlib.pyplot as plt
    lens = [s.shape[0] for s in all_seq]
    plt.hist(lens, bins=50)
    plt.axvline(300, color='red', linestyle='--')
    plt.title("NTU RGB+D sequence length distribution")
    plt.show()
    plt.savefig("ntu_seq_length_distribution.png")
    t_end = time.time()
    print(f"[INFO] Time taken to load NTU skeletons: {t_end - t_start:.2f} seconds")



if __name__ == "__main__":
    main()