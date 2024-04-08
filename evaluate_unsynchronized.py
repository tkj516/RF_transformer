import os
from typing import Callable

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from absl import app, flags
from ml_collections import ConfigDict
from tqdm import tqdm

import rfcutils2 as rfcutils
from models.transformer import Transformer
from utils.rf_dataset import RFDatasetBase
from torch.utils.data import DataLoader

tf.config.set_visible_devices([], "GPU")


# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "soi_root_dir", default="", help="Root directory for the SOI dataset"
)
flags.DEFINE_string(
    "interference_root_dir",
    default="",
    help="Root directory for the interference dataset",
)
flags.DEFINE_string(
    "checkpoint_dir", default="", help="Path to the checkpoint directory"
)
flags.DEFINE_string("soi_type", default="QPSK", help="Type of SOI")
flags.DEFINE_string(
    "interference_sig_type", default="CommSignal2", help="Type of interference signal"
)
flags.DEFINE_string(
    "testset_identifier", default="TestSet1Mixture", help="Identifier of the testset"
)
flags.DEFINE_string(
    "id_string", default="transformer", help="Identifier of the output files"
)
flags.DEFINE_integer("batch_size", default=10, help="Batch size for inference")


ALL_SINR = np.arange(-30, 0.1, 3)


class UnsynchronizedRFTestDataset(RFDatasetBase):
    def __init__(
        self,
        soi_root_dir: str,
        interference_root_dir: str,
        window_size: int,
        context_size: int,
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

        self.interference_files = self.interference_files[:len(self.soi_files)]

        assert (
            signal_length % window_size == 0
        ), "Signal length should be multiple of window size"

        self.signal_length = signal_length
        self.number_soi_offsets = number_soi_offsets
        self.use_rand_phase = use_rand_phase

    def __len__(self):
        return len(self.soi_files)

    def __getitem__(self, idx):
        soi = np.load(self.soi_files[idx])
        interference = np.load(self.interference_files[idx])

        rand_soi_start_idx = np.random.randint(self.number_soi_offsets)
        rand_interference_start_idx = np.random.randint(
            len(interference) - self.signal_length
        )

        soi = soi[rand_soi_start_idx : rand_soi_start_idx + self.signal_length]
        interference = interference[
            rand_interference_start_idx : rand_interference_start_idx
            + self.signal_length
        ]

        return {
            "soi": soi,
            "interference": interference,
            "soi_offset": rand_soi_start_idx,
        }


def get_soi_demod_fn(soi_sig_type: str, sig_len: int = 40960) -> Callable:
    if soi_sig_type == "QPSK":
        demod_soi = rfcutils.qpsk_matched_filter_demod
    elif soi_sig_type == "OFDMQPSK":
        _, _, _, RES_GRID = rfcutils.generate_ofdm_signal(1, sig_len // 80)

        def demod_soi(s):
            return rfcutils.ofdm_demod(s, RES_GRID)
    else:
        raise Exception("SOI Type not recognized")
    return demod_soi


def process_inputs(
    soi: torch.Tensor,
    interference: torch.Tensor,
    sinr_db: float,
    window_size: int,
    context_size: int,
    use_rand_phase: bool = True,
) -> torch.Tensor:
    coeff = 10 ** (-0.5 * sinr_db / 10)
    if use_rand_phase:
        rand_phase = np.random.rand()
        coeff = coeff * np.exp(1j * 2 * np.pi * rand_phase)
    mixture = soi + coeff * interference

    soi = torch.view_as_real(soi).to(torch.float32)
    mixture = torch.view_as_real(mixture).to(torch.float32)

    soi_target = soi.unfold(1, window_size, window_size).reshape(
        soi.shape[0], -1, window_size * 2
    )

    soi = F.pad(soi, (0, 0, context_size, 0, 0, 0))
    mixture = F.pad(mixture, (0, 0, context_size, 0, 0, 0))
    soi_windows = soi.unfold(1, context_size + window_size, window_size).reshape(
        soi.shape[0], -1, (window_size + context_size) * 2
    )
    mixture_windows = mixture.unfold(
        1, context_size + window_size, window_size
    ).reshape(soi.shape[0], -1, (window_size + context_size) * 2)

    assert soi_windows.shape[0] == soi_target.shape[0], (
        soi_windows.shape,
        soi_target.shape,
    )

    return {
        "soi": soi_windows,
        "mixture": mixture_windows,
        "target": soi_target,
    }


@torch.no_grad()
def run_inference(
    soi_root_dir: str,
    interference_root_dir: str,
    checkpoint_dir: str,
    soi_type: str,
    device: torch.device,
    batch_size: int = 32,
):
    demod_soi = get_soi_demod_fn(soi_type)

    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
    checkpoint = torch.load(checkpoint_dir, map_location="cpu")
    config = ConfigDict(checkpoint["cfg"])
    model = Transformer(**config.model_config[1]).to(device)

    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError:
        new_state_dict = {}
        for k, v in checkpoint["model"].items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)

    window_size = config.dataset_config[1]["window_size"]
    context_size = config.dataset_config[1]["context_size"]

    dataset = UnsynchronizedRFTestDataset(
        soi_root_dir=soi_root_dir,
        interference_root_dir=interference_root_dir,
        window_size=window_size,
        context_size=context_size,
        signal_length=40960,
        number_soi_offsets=16,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    all_soi_est = []
    all_soi_gt = []
    for sinr_db in ALL_SINR:
        for inputs in tqdm(dataloader):
            soi = inputs["soi"]
            interference = inputs["interference"]

            processed = process_inputs(
                soi,
                interference,
                sinr_db,
                window_size,
                context_size,
            )

            soi = processed["soi"]
            mixture = processed["mixture"]
            soi = model.embed_patch(soi.to(device))
            mixture = model.embed_patch(mixture.to(device))

            soi_est = model.generate(
                cond=mixture,
                window_size=window_size,
                context_size=context_size,
            )
            waveform = (
                soi_est.cpu()
                .reshape(*soi_est.shape[:2], 2, window_size)
                .permute(0, 1, 3, 2)
                .reshape(soi_est.shape[0], -1, 2)
            )
            waveform = torch.view_as_complex(waveform)

            slice_idx = inputs["soi_offset"].reshape(-1, 1) + torch.arange(
                40960 - 16,
                device=inputs["soi_offset"].device,
            ).reshape(1, -1)
            waveform = torch.take_along_dim(
                waveform, slice_idx.to(waveform.device), dim=1
            )
            gt_waveform = torch.take_along_dim(
                inputs["soi"], slice_idx.to(inputs["soi"].device), dim=1
            )
            all_soi_est.append(waveform.numpy())
            all_soi_gt.append(gt_waveform.numpy())
    soi_est = np.concatenate(all_soi_est, axis=0)
    soi_gt = np.concatenate(all_soi_gt, axis=0)

    bit_est = []
    bit_gt = []
    for est, gt in zip(np.split(soi_est, batch_size), np.split(soi_gt, batch_size)):
        bit_est_batch, _ = demod_soi(est)
        bit_gt_batch, _ = demod_soi(gt)
        bit_est.append(bit_est_batch.numpy())
        bit_gt.append(bit_gt_batch.numpy())
    bit_est = np.concatenate(bit_est, axis=0)
    bit_gt = np.concatenate(bit_gt, axis=0)
    return soi_gt, bit_gt, soi_est, bit_est


def main(_):
    testset_identifier = FLAGS.testset_identifier
    id_string = FLAGS.id_string

    sig1_gt, bit1_gt, sig1_est, bit1_est = run_inference(
        FLAGS.soi_root_dir,
        FLAGS.interference_root_dir,
        FLAGS.checkpoint_dir,
        FLAGS.soi_type,
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        FLAGS.batch_size,
    )

    if not os.path.exists("eval_outputs"):
        os.makedirs("eval_outputs")

    np.save(
        os.path.join(
            "eval_outputs",
            "unsynchronized",
            f"{id_string}_{testset_identifier}_gt_soi_{FLAGS.soi_type}"
            f"_{FLAGS.interference_sig_type}",
        ),
        sig1_gt,
    )
    np.save(
        os.path.join(
            "eval_outputs",
            "unsynchronized",
            f"{id_string}_{testset_identifier}_gt_bits_{FLAGS.soi_type}"
            f"_{FLAGS.interference_sig_type}",
        ),
        bit1_gt,
    )
    np.save(
        os.path.join(
            "eval_outputs",
            "unsynchronized",
            f"{id_string}_{testset_identifier}_estimated_soi_{FLAGS.soi_type}"
            f"_{FLAGS.interference_sig_type}",
        ),
        sig1_est,
    )
    np.save(
        os.path.join(
            "eval_outputs",
            "unsynchronized",
            f"{id_string}_{testset_identifier}_estimated_bits_{FLAGS.soi_type}"
            f"_{FLAGS.interference_sig_type}",
        ),
        bit1_est,
    )


if __name__ == "__main__":
    app.run(main)
