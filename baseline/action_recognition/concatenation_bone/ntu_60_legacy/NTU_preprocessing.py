import os
import glob
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

NUM_JOINTS = 25
MAX_FRAMES = 300
SKELETON_ROOT = 'nturgb+d_skeletons'

XSUB_TRAIN_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
XVIEW_TRAIN_CAMERAS = [2, 3]


def read_skeleton(filepath):
    with open(filepath, "r") as f:
        n_frames = int(f.readline())
        data = np.zeros((n_frames, NUM_JOINTS, 3), dtype=np.float32)

        for t in range(n_frames):
            n_bodies = int(f.readline())
            if n_bodies == 0:
                continue

            f.readline()  # bodyID
            f.readline()  # body info line

            for j in range(NUM_JOINTS):
                values = list(map(float, f.readline().split()))
                data[t, j] = values[:3]

            for _ in range(n_bodies - 1):
                f.readline()
                f.readline()
                for _ in range(NUM_JOINTS):
                    f.readline()

    # Center by hip joint (joint 0)
    hip = data[:, 0:1, :]
    return data - hip


def pad_sequence(seq, max_len=MAX_FRAMES):
    T, V, C = seq.shape
    if T >= max_len:
        return seq[:max_len]
    else:
        pad = np.zeros((max_len - T, V, C), dtype=np.float32)
        return np.concatenate([seq, pad], axis=0)


def preprocess(split='xsub'):
    train_data, train_labels = [], []
    test_data, test_labels = [], []

    for filepath in tqdm(sorted(glob.glob(os.path.join(SKELETON_ROOT, '*.skeleton')))):
        filename = os.path.basename(filepath)
        subject_id = int(filename[9:12])
        camera_id = int(filename[5:8])
        action_id = int(filename[17:20]) - 1  # zero-based

        data = read_skeleton(filepath)             # (T, 25, 3)
        data = pad_sequence(data, MAX_FRAMES)      # (MAX_FRAMES, 25, 3)

        # FIXME: encode one person for now!
        padded = data

        if split == 'xsub':
            if subject_id in XSUB_TRAIN_SUBJECTS:
                train_data.append(padded)
                train_labels.append(action_id)
            else:
                test_data.append(padded)
                test_labels.append(action_id)

        elif split == 'xview':
            if camera_id in XVIEW_TRAIN_CAMERAS:
                train_data.append(padded)
                train_labels.append(action_id)
            else:
                test_data.append(padded)
                test_labels.append(action_id)

    # Convert to numpy arrays
    x_train = np.array(train_data)
    x_test = np.array(test_data)

    y_train = np.array(train_labels).reshape(-1, 1)
    y_test = np.array(test_labels).reshape(-1, 1)

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train)
    y_test_onehot = encoder.transform(y_test)

    return x_train, y_train_onehot, x_test, y_test_onehot


if __name__ == '__main__':
    for split in ['xsub', 'xview']:
        print(f"\n[INFO] Processing split: {split}")
        x_train, y_train, x_test, y_test = preprocess(split=split)

        print(f"[INFO] Train: {x_train.shape}, Test: {x_test.shape}")
        print(f"[INFO] Classes: {y_train.shape[1]}")

        np.savez(f'NTU_SF_{split}.npz',
                 x_train=x_train,
                 y_train=y_train,
                 x_test=x_test,
                 y_test=y_test)

        print(f"✅ [INFO] Saved: NTU_SF_{split}.npz")
