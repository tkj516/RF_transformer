import glob
import os
from typing import Tuple, Union

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
        context_size: Union[int, Tuple[int, int]],
    ):
        self.soi_root_dir = soi_root_dir
        self.interference_root_dir = interference_root_dir
        self.window_size = window_size

        if isinstance(context_size, int):
            self.left_context_size = context_size
            self.right_context_size = 0
        else:
            self.left_context_size = context_size[0]
            self.right_context_size = context_size[1]

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

        soi = F.pad(soi, (0, 0, self.left_context_size, self.right_context_size))
        interference = F.pad(
            interference, (0, 0, self.left_context_size, self.right_context_size)
        )
        soi_windows = soi.unfold(
            0,
            self.left_context_size + self.window_size + self.right_context_size,
            self.window_size,
        ).reshape(
            -1,
            (self.left_context_size + self.window_size + self.right_context_size) * 2,
        )
        interference_windows = interference.unfold(
            0,
            self.left_context_size + self.window_size + self.right_context_size,
            self.window_size,
        ).reshape(
            -1,
            (self.left_context_size + self.window_size + self.right_context_size) * 2,
        )

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


class ICASSPDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        window_size: int,
        context_size: int,
    ):
        self.root_dir = root_dir
        self.window_size = window_size
        self.context_size = context_size

        self.files = glob.glob(os.path.join(root_dir, "*.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npy_file = np.load(self.files[idx], allow_pickle=True).item()
        soi = torch.from_numpy(npy_file["sample_soi"]).to(torch.float32)
        mixture = torch.from_numpy(npy_file["sample_mix"]).to(torch.float32)

        sequence_length = (mixture.shape[0] // self.window_size) * self.window_size

        soi = soi[:sequence_length]
        mixture = mixture[:sequence_length]

        soi_target = soi.unfold(0, self.window_size, self.window_size).reshape(
            -1, self.window_size * 2
        )

        soi = F.pad(soi, (0, 0, self.context_size, 0))
        mixture = F.pad(mixture, (0, 0, self.context_size, 0))
        soi_windows = soi.unfold(
            0, self.context_size + self.window_size, self.window_size
        ).reshape(-1, (self.window_size + self.context_size) * 2)
        mixture_windows = mixture.unfold(
            0, self.context_size + self.window_size, self.window_size
        ).reshape(-1, (self.window_size + self.context_size) * 2)

        assert soi_windows.shape[0] == soi_target.shape[0], (
            soi_windows.shape,
            soi_target.shape,
        )

        return {
            "soi": soi_windows,
            "mixture": mixture_windows,
            "target": soi_target,
        }


class UnsynchronizedRFDataset(RFDatasetBase):
    def __init__(
        self,
        soi_root_dir: str,
        interference_root_dir: str,
        window_size: int,
        context_size: Union[int, Tuple[int, int]],
        signal_length: int,
        number_soi_offsets: int,
        use_rand_phase: bool = True,
    ):
        super().__init__(
            soi_root_dir=soi_root_dir,
            interference_root_dir=interference_root_dir,
            window_size=window_size,
            context_size=context_size,
        )

        assert (
            signal_length % window_size == 0
        ), "Signal length should be multiple of window size"

        self.signal_length = signal_length
        self.number_soi_offsets = number_soi_offsets
        self.use_rand_phase = use_rand_phase

    def __len__(self):
        return len(self.soi_files)

    def __getitem__(self, idx):
        soi_idx = idx
        interference_idx = np.random.randint(len(self.interference_files))

        soi = np.load(self.soi_files[soi_idx])
        interference = np.load(self.interference_files[interference_idx])

        rand_soi_start_idx = np.random.randint(self.number_soi_offsets)
        rand_interference_start_idx = np.random.randint(
            len(interference) - self.signal_length
        )

        soi = soi[rand_soi_start_idx : rand_soi_start_idx + self.signal_length]
        interference = interference[
            rand_interference_start_idx : rand_interference_start_idx
            + self.signal_length
        ]

        sinr_db = -36 * np.random.rand() + 3
        coeff = 10 ** (-0.5 * sinr_db / 10)
        if self.use_rand_phase:
            rand_phase = np.random.rand()
            coeff = coeff * np.exp(1j * 2 * np.pi * rand_phase)
        mixture = soi + coeff * interference

        soi = torch.view_as_real(torch.from_numpy(soi)).to(torch.float32)
        mixture = torch.view_as_real(torch.from_numpy(mixture)).to(torch.float32)

        soi_target = soi.unfold(0, self.window_size, self.window_size).reshape(
            -1, self.window_size * 2
        )

        soi = F.pad(soi, (0, 0, self.left_context_size, self.right_context_size))
        mixture = F.pad(
            mixture, (0, 0, self.left_context_size, self.right_context_size)
        )
        soi_windows = soi.unfold(
            0,
            self.left_context_size + self.window_size + self.right_context_size,
            self.window_size,
        ).reshape(
            -1,
            (self.left_context_size + self.window_size + self.right_context_size) * 2,
        )
        mixture_windows = mixture.unfold(
            0,
            self.left_context_size + self.window_size + self.right_context_size,
            self.window_size,
        ).reshape(
            -1,
            (self.left_context_size + self.window_size + self.right_context_size) * 2,
        )

        assert soi_windows.shape[0] == soi_target.shape[0], (
            soi_windows.shape,
            soi_target.shape,
        )

        return {
            "soi": soi_windows,
            "mixture": mixture_windows,
            "target": soi_target,
            "soi_offset": rand_soi_start_idx,
        }


if __name__ == "__main__":
    dataset = RFDatasetBase(
        soi_root_dir="/home/tejasj/data2/RF_transformer/dataset/qpsk/qpsk_200000_160",
        interference_root_dir="/home/tejasj/data2/RF_transformer/dataset/ofdm/ofdm_200000_32",
        window_size=128,
        context_size=32,
    )

    dataset[0]
