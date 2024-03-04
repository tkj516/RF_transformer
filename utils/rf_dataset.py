import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class RFDatasetBase(Dataset):
    def __init__(
        self,
        soi_root_dir: str,
        interference_root_dir: str,
        window_size: int,
        context_size: int,
    ):
        self.soi_root_dir = soi_root_dir
        self.interference_root_dir = interference_root_dir
        self.window_size = window_size
        self.context_size = context_size

        self.soi_files = glob.glob(os.path.join(soi_root_dir, "*.npy"))
        self.interference_files = glob.glob(
            os.path.join(interference_root_dir, "*.npy")
        )

    def __len__(self):
        return len(self.soi_files)

    def __getitem__(self, idx):
        soi = torch.view_as_real(torch.from_numpy(np.load(self.soi_files[idx]))).to(
            torch.float32
        )
        interference = torch.view_as_real(
            torch.from_numpy(np.load(self.interference_files[idx]))
        ).to(torch.float32)

        min_length = min(soi.shape[0], interference.shape[0])
        sequence_length = (min_length // self.window_size) * self.window_size

        soi = soi[:sequence_length]
        interference = interference[:sequence_length]

        soi_target = soi.unfold(0, self.window_size, self.window_size).reshape(
            -1, self.window_size * 2
        )

        soi = F.pad(soi, (0, 0, self.context_size, 0))
        interference = F.pad(interference, (0, 0, self.context_size, 0))
        soi_windows = soi.unfold(
            0, self.context_size + self.window_size, self.window_size
        ).reshape(-1, (self.window_size + self.context_size) * 2)
        interference_windows = interference.unfold(
            0, self.context_size + self.window_size, self.window_size
        ).reshape(-1, (self.window_size + self.context_size) * 2)

        assert (
            soi_windows.shape[0] == interference_windows.shape[0] == soi_target.shape[0]
        ), (
            soi_windows.shape,
            interference_windows.shape,
            soi_target.shape,
        )

        sinr_db = -36 * np.random.rand() + 3
        coeff = 10 ** (-0.5 * sinr_db / 10)
        mixture_windows = soi_windows + coeff * interference_windows

        return {
            "soi": soi_windows,
            "interference": interference_windows,
            "mixture": mixture_windows,
            "target": soi_target,
        }


if __name__ == "__main__":
    dataset = RFDatasetBase(
        soi_root_dir="/home/tejasj/data2/RF_transformer/dataset/qpsk/qpsk_200000_160",
        interference_root_dir="/home/tejasj/data2/RF_transformer/dataset/ofdm/ofdm_200000_32",
        window_size=128,
        context_size=32,
    )

    dataset[0]
