import os
import sys
import glob
import h5py
import numpy as np
from tqdm import tqdm

def preprocess_dataset(root_dir: str, save_dir: str) -> None:
    save_dir = os.path.join(save_dir, os.path.basename(root_dir))
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    for folder in tqdm(glob.glob(os.path.join(root_dir, "*.h5"))):
        with h5py.File(folder, "r") as f:
            mixture = np.array(f.get("mixture"))
            soi = np.array(f.get("target"))
        for i in range(mixture.shape[0]):
            data = {
                "sample_mix": mixture[i, ...],
                "sample_soi": soi[i, ...],
            }
            np.save(os.path.join(save_dir, f"sample_{count}.npy"), data)
            count += 1


if __name__ == "__main__":
    dataset_type = sys.argv[1]
    preprocess_dataset(root_dir=f'dataset/Dataset_{dataset_type}_Mixture', 
                       save_dir='npydataset/')