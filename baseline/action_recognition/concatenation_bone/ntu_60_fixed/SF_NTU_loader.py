import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import glob

NUM_JOINTS = 25
MAX_FRAMES = 300
SKELETON_ROOT = 'nturgb+d_skeletons'

XSUB_TRAIN_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
XVIEW_TRAIN_CAMERAS = [2, 3]

class SF_NTU_Dataset(Dataset):
    def __init__(self, data_path, label_path=None, split='train', window_size=64, thres=64, aug_method=''):
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.window_size = window_size
        self.thres = thres
        self.aug_method = aug_method
        self.load_data()

    def load_data(self):
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']  # (N, T, V, C)
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = [f'train_{i}' for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = [f'test_{i}' for i in range(len(self.data))]
        else:
            raise ValueError("Split must be 'train' or 'test'")

    def __len__(self):
        return len(self.label)


    def __getitem__(self, index):
        data_numpy = self.data[index]  # (T, V, C)
        print(f"[DEBUG] Data shape: {data_numpy.shape}")
        label = self.label[index]

        valid_frame_num = np.sum(np.any(data_numpy != 0, axis=(1, 2)))
        data_numpy = data_numpy[:valid_frame_num]

        if valid_frame_num < self.thres:
            padded = np.zeros((self.window_size, NUM_JOINTS, 3), dtype=np.float32)
            index_t = np.linspace(-1, 1, self.window_size)

            padded = padded.reshape(self.window_size, -1)

            return padded, index_t, label, index

        data_numpy = self.crop_and_resize(data_numpy, self.window_size)
        data_numpy = self.apply_augmentation(data_numpy)

        index_t = np.linspace(-1, 1, self.window_size)
        
        data_numpy = data_numpy.reshape(self.window_size, -1)
        
        return data_numpy.astype(np.float32), index_t.astype(np.float32), label, index

    def crop_and_resize(self, data_numpy, target_len):
        T = data_numpy.shape[0]
        if T == target_len:
            return data_numpy
        elif T > target_len:
            start = random.randint(0, T - target_len)
            return data_numpy[start:start + target_len]
        else:
            pad = np.zeros((target_len - T, data_numpy.shape[1], data_numpy.shape[2]), dtype=np.float32)
            return np.concatenate([data_numpy, pad], axis=0)

    def apply_augmentation(self, data_numpy):
        if '1' in self.aug_method and random.random() < 0.5:
            shear = [random.uniform(-0.5, 0.5) for _ in range(6)]
            R = np.array([[1, shear[0], shear[1]],
                          [shear[2], 1, shear[3]],
                          [shear[4], shear[5], 1]])
            data_numpy = np.dot(data_numpy.reshape(-1, 3), R).reshape(data_numpy.shape)

        if '2' in self.aug_method and random.random() < 0.5:
            angle = random.uniform(-30, 30) * np.pi / 180
            axis = random.choice(['x', 'y', 'z'])
            if axis == 'x':
                R = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
            elif axis == 'y':
                R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            else:
                R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            data_numpy = np.dot(data_numpy.reshape(-1, 3), R).reshape(data_numpy.shape)

        if '3' in self.aug_method and random.random() < 0.5:
            scale_factor = np.random.uniform(0.8, 1.2)
            data_numpy *= scale_factor

        if '4' in self.aug_method and random.random() < 0.5:
            # spatial flip — swap left and right joints
            flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]
            data_numpy = data_numpy[:, flip_index, :]

        if '5' in self.aug_method and random.random() < 0.5:
            data_numpy = data_numpy[::-1]

        if '6' in self.aug_method and random.random() < 0.5:
            data_numpy += np.random.normal(0, 0.05, size=data_numpy.shape)

        if '9' in self.aug_method and random.random() < 0.5:
            joint_idx = np.random.choice(data_numpy.shape[1], size=3, replace=False)
            data_numpy[:, joint_idx] = 0

        return data_numpy


def read_skeleton(filepath):
    with open(filepath, "r") as f:
        n_frames = int(f.readline())
        data = np.zeros((n_frames, NUM_JOINTS, 3), dtype=np.float32)

        for t in range(n_frames):
            n_bodies = int(f.readline())
            if n_bodies == 0:
                continue

            f.readline()
            f.readline()

            for j in range(NUM_JOINTS):
                values = list(map(float, f.readline().split()))
                data[t, j] = values[:3]

            for _ in range(n_bodies - 1):
                f.readline()
                f.readline()
                for _ in range(NUM_JOINTS):
                    f.readline()

    hip = data[:, 0:1, :]
    return data - hip


def pad_sequence(seq, max_len=MAX_FRAMES):
    T, V, C = seq.shape
    if T >= max_len:
        return seq[:max_len]
    else:
        pad = np.zeros((max_len - T, V, C), dtype=np.float32)
        return np.concatenate([seq, pad], axis=0)

if __name__ == "__main__":
    # Example usage
    dataset = SF_NTU_Dataset(data_path='NTU_SF_xsub.npz', split='train', window_size=64, aug_method='')
    for i in range(len(dataset)):
        data, index_t, label, idx = dataset[i]
        print(f"Data shape: {data.shape}, Index T shape: {index_t.shape}, Label: {label}, Index: {idx}")
        if i == 5:  # Just to limit the output
            break