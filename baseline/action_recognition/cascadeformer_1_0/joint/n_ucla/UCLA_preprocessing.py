import os
import glob
import numpy as np
import pickle
import json
from tqdm import tqdm

def process_nucla_to_json(root="N_UCLA", save_dir="N-UCLA_processed"):
    os.makedirs(save_dir, exist_ok=True)

    label_map = {}  # e.g., {'a01': 1, 'a02': 2, ...}
    label_counter = 1

    datasets = {
        "train": {"views": ["view_1", "view_2"], "meta": []},
        "val":   {"views": ["view_3"], "meta": []}
    }

    for split in ["train", "val"]:
        for view in datasets[split]["views"]:
            view_idx = int(view.split("_")[-1])
            view_path = os.path.join(root, view)
            sample_dirs = sorted(glob.glob(os.path.join(view_path, "a*_s*_e*")))

            for sample_path in tqdm(sample_dirs, desc=f"[{split}] {view}"):
                sample_name = os.path.basename(sample_path)
                file_name = f"{sample_name}_v0{view_idx}"  # expected .json name

                action = sample_name.split('_')[0]
                if action not in label_map:
                    label_map[action] = label_counter
                    label_counter += 1
                label = label_map[action]

                # read frames
                skeleton_files = sorted(glob.glob(os.path.join(sample_path, "*_skeletons.txt")))
                frames = []
                for f in skeleton_files:
                    with open(f, 'r') as fin:
                        lines = fin.readlines()
                    if len(lines) < 21:
                        continue
                    joints = []
                    for line in lines[1:21]:
                        x, y, z, _ = map(float, line.strip().split(','))
                        joints.append([x, y, z])
                    frames.append(joints)

                if len(frames) == 0:
                    continue

                skeleton_arr = np.array(frames, dtype=np.float32)  # (T, 20, 3)

                # Save to individual JSON file
                json_dict = {"skeletons": skeleton_arr.tolist()}
                with open(os.path.join(save_dir, f"{file_name}.json"), "w") as jf:
                    json.dump(json_dict, jf)

                # Append metadata
                datasets[split]["meta"].append({
                    "file_name": file_name,
                    "length": len(frames),
                    "label": label  # one-indexed
                })

    for split in ["train", "val"]:
        label_list = [item["label"] for item in datasets[split]["meta"]]
        with open(os.path.join(save_dir, f"{split}_label.pkl"), "wb") as f:
            pickle.dump((label_list, list(label_map.keys())), f)
        with open(os.path.join(save_dir, f"{split}_metadata.json"), "w") as f:
            json.dump(datasets[split]["meta"], f, indent=2)

        print(f"[INFO] Wrote {len(datasets[split]['meta'])} samples to '{save_dir}' as JSON.")
    print(f"[INFO] Total classes: {len(label_map)}")

if __name__ == "__main__":
    process_nucla_to_json()
