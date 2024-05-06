import sys

sys.path.append("..")

import glob
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
from models.transformer_decoder import Transformer as TransformerDecoder
from models.wavenet import Wave
from torch.utils.data import Dataset, DataLoader
from utils.dataset import get_soi_generation_fn

# tf.config.set_visible_devices([], "GPU")


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
flags.DEFINE_bool("matched_filter_only", default=False, help="Run matched filter only")
flags.DEFINE_bool("lmmse", default=False, help="Run LMMSE")
flags.DEFINE_bool(
    "decoder_only", default=False, help="Use decoder only transformer model."
)
flags.DEFINE_bool("wavenet", default=False, help="Use wavenet model.")


ALL_SINR = np.arange(-30, 0.1, 3)


class UnsynchronizedRFTestDataset(Dataset):
    def __init__(
        self,
        soi_root_dir: str,
        interference_root_dir: str,
        signal_length: int,
        number_soi_offsets: int,
    ):
        super().__init__()

        self.soi_root_dir = soi_root_dir
        self.interference_root_dir = interference_root_dir

        self.soi_files = glob.glob(os.path.join(soi_root_dir, "*.npy"))
        self.interference_files = glob.glob(
            os.path.join(interference_root_dir, "*.npy")
        )
        self.interference_files = self.interference_files[: len(self.soi_files)]

        self.signal_length = signal_length
        self.number_soi_offsets = number_soi_offsets

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

    if isinstance(context_size, int):
        left_context_size = context_size
        right_context_size = 0
    else:
        left_context_size = context_size[0]
        right_context_size = context_size[1]

    soi = F.pad(soi, (0, 0, left_context_size, right_context_size, 0, 0))
    mixture = F.pad(mixture, (0, 0, left_context_size, right_context_size, 0, 0))
    soi_windows = soi.unfold(
        1, left_context_size + window_size + right_context_size, window_size
    ).reshape(
        soi.shape[0], -1, (left_context_size + window_size + right_context_size) * 2
    )
    mixture_windows = mixture.unfold(
        1, left_context_size + window_size + right_context_size, window_size
    ).reshape(
        soi.shape[0], -1, (left_context_size + window_size + right_context_size) * 2
    )

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
    decoder_only: bool = False,
    wavenet: bool = False,
):
    demod_soi = get_soi_demod_fn(soi_type)

    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
    checkpoint = torch.load(checkpoint_dir, map_location="cpu")
    config = ConfigDict(checkpoint["cfg"])

    if decoder_only:
        model = TransformerDecoder(**config.model_config[1]).to(device)
    elif wavenet:
        model = Wave(**config.model_config[1]).to(device)
    else:
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

    dataset = UnsynchronizedRFTestDataset(
        soi_root_dir=soi_root_dir,
        interference_root_dir=interference_root_dir,
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

            if FLAGS.wavenet:
                coeff = 10 ** (-0.5 * sinr_db / 10)
                rand_phase = np.random.rand()
                coeff = coeff * np.exp(1j * 2 * np.pi * rand_phase)
                mixture = soi + coeff * interference
                
                soi_est = model(
                    torch.view_as_real(mixture)
                    .to(torch.float32)
                    .permute(0, 2, 1)
                    .contiguous()
                    .to(device)
                )
                waveform = torch.view_as_complex(
                    soi_est.cpu().permute(0, 2, 1).contiguous()
                )
            else:
                window_size = config.dataset_config[1]["window_size"]
                context_size = config.dataset_config[1]["context_size"]

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

                kwargs = {}
                if decoder_only:
                    kwargs["input"] = mixture
                else:
                    kwargs["cond"] = mixture
                    kwargs["window_size"] = window_size
                    kwargs["context_size"] = context_size
                soi_est = model.generate(**kwargs)
                waveform = (
                    soi_est.cpu()
                    .reshape(*soi_est.shape[:2], 2, window_size)
                    .permute(0, 1, 3, 2)
                    .reshape(soi_est.shape[0], -1, 2)
                )
                waveform = torch.view_as_complex(waveform)

            slice_idx = (16 - inputs["soi_offset"]).reshape(-1, 1) + torch.arange(
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


@torch.no_grad()
def run_matched_filter(
    soi_root_dir: str,
    interference_root_dir: str,
    soi_type: str,
    batch_size: int = 32,
):
    demod_soi = get_soi_demod_fn(soi_type)

    dataset = UnsynchronizedRFTestDataset(
        soi_root_dir=soi_root_dir,
        interference_root_dir=interference_root_dir,
        signal_length=40960,
        number_soi_offsets=16,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    all_soi_est = []
    for sinr_db in ALL_SINR:
        for inputs in tqdm(dataloader):
            soi = inputs["soi"]
            interference = inputs["interference"]

            coeff = 10 ** (-0.5 * sinr_db / 10)
            rand_phase = np.random.rand()
            coeff = coeff * np.exp(1j * 2 * np.pi * rand_phase)
            mixture = soi + coeff * interference

            slice_idx = (16 - inputs["soi_offset"]).reshape(-1, 1) + torch.arange(
                40960 - 16,
                device=inputs["soi_offset"].device,
            ).reshape(1, -1)
            waveform = torch.take_along_dim(
                mixture, slice_idx.to(mixture.device), dim=1
            )
            all_soi_est.append(waveform.numpy())
    soi_est = np.concatenate(all_soi_est, axis=0)

    bit_est = []
    for est in np.split(soi_est, batch_size):
        bit_est_batch, _ = demod_soi(est)
        bit_est.append(bit_est_batch.numpy())
    bit_est = np.concatenate(bit_est, axis=0)
    return soi_est, bit_est


def estimate_covariance(generate_soi: Callable, interference_root_dir: str):
    all_interference_sig = []
    for interference_file in glob.glob(os.path.join(interference_root_dir, "*.npy")):
        interference = np.load(interference_file)
        all_interference_sig.append(interference.reshape(1, -1))
    all_interference = np.concatenate(all_interference_sig, axis=0)

    cov_sig_len = 2559
    with tf.device("CPU"):
        all_sig1, _, _, _ = generate_soi(10000, cov_sig_len)
        all_sig1 = tf.reshape(all_sig1, (-1, cov_sig_len))
        all_sig1_T = tf.transpose(all_sig1)
        cov_s = (
            1
            / all_sig1.shape[0]
            * tf.linalg.matmul(all_sig1_T, all_sig1_T, adjoint_b=True)
        )
        cov_s = cov_s.numpy()

        all_interference = all_interference[
            np.random.randint(all_interference.shape[0], size=(10000)), :
        ]
        rand_start_idx = np.random.randint(
            all_interference.shape[1] - 40960 + 16, size=all_interference.shape[0]
        )
        indices = tf.cast(
            rand_start_idx.reshape(-1, 1) + np.arange(40960 - 16).reshape(1, -1),
            tf.int32,
        )
        all_interference = tf.experimental.numpy.take_along_axis(
            all_interference, indices, axis=1
        )

        all_interference = tf.reshape(all_interference, (-1, cov_sig_len))
        all_interference_T = tf.transpose(all_interference)
        cov_b = (
            1
            / all_interference.shape[0]
            * tf.linalg.matmul(all_interference_T, all_interference_T, adjoint_b=True)
        )
        cov_b = cov_b.numpy()

    return cov_s, cov_b, cov_sig_len


def run_lmmse(
    soi_root_dir: str,
    covariance_interference_root_dir: str,
    interference_root_dir: str,
    soi_type: str,
    batch_size: int = 32,
):
    generate_soi = get_soi_generation_fn(soi_type)
    demod_soi = get_soi_demod_fn(soi_type)

    dataset = UnsynchronizedRFTestDataset(
        soi_root_dir=soi_root_dir,
        interference_root_dir=interference_root_dir,
        signal_length=40960,
        number_soi_offsets=16,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    cov_s, cov_b, cov_sig_len = estimate_covariance(
        generate_soi, covariance_interference_root_dir
    )
    soi_est = []
    for sinr_db in tqdm(enumerate(ALL_SINR)):
        for inputs in tqdm(dataloader):
            soi = inputs["soi"]
            interference = inputs["interference"]

            coeff = 10 ** (-0.5 * sinr_db / 10)
            rand_phase = np.random.rand()
            coeff = coeff * np.exp(1j * 2 * np.pi * rand_phase)
            mixture = soi + coeff * interference

            slice_idx = (16 - inputs["soi_offset"]).reshape(-1, 1) + torch.arange(
                40960 - 16,
                device=inputs["soi_offset"].device,
            ).reshape(1, -1)
            waveform = torch.take_along_dim(
                mixture, slice_idx.to(mixture.device), dim=1
            )

            mixture_reshaped = waveform.numpy().reshape(-1, cov_sig_len)
            sinr = 10 ** (sinr_db / 10)
            Cyy = cov_s + 1 / sinr * cov_b
            Csy = cov_s.copy()

            U, S, _ = np.linalg.svd(Cyy, hermitian=True)
            sthr_idx = np.linalg.matrix_rank(Cyy) + 1
            Cyy_inv = np.matmul(
                U[:, :sthr_idx],
                np.matmul(np.diag(1.0 / (S[:sthr_idx])), U[:, :sthr_idx].conj().T),
            )
            # Cyy_inv = np.linalg.pinv(Cyy)
            W = np.matmul(Csy, Cyy_inv)

            lmmse = np.matmul(W, mixture_reshaped.T).T
            soi_est.append(np.reshape(lmmse, (-1, 40960 - 16)))
    soi_est = np.concatenate(soi_est, axis=0).astype(np.complex64)

    bit_est = []
    for est in np.split(soi_est, batch_size):
        bit_est_batch, _ = demod_soi(est)
        bit_est.append(bit_est_batch.numpy())
    bit_est = np.concatenate(bit_est, axis=0)
    return soi_est, bit_est


def main(_):
    testset_identifier = FLAGS.testset_identifier
    id_string = FLAGS.id_string

    if not os.path.exists("/home/tejasj/data2/RF_transformer/eval_outputs"):
        os.makedirs("/home/tejasj/data2/RF_transformer/eval_outputs")

    if FLAGS.lmmse:
        id_string = "lmmse"

        sig1_est, bit1_est = run_lmmse(
            FLAGS.soi_root_dir,
            FLAGS.interference_root_dir,
            FLAGS.interference_root_dir,
            FLAGS.soi_type,
            FLAGS.batch_size,
        )
        np.save(
            os.path.join(
                "/home/tejasj/data2/RF_transformer/eval_outputs",
                "unsynchronized",
                f"{id_string}_{testset_identifier}_estimated_soi_{FLAGS.soi_type}"
                f"_{FLAGS.interference_sig_type}",
            ),
            sig1_est,
        )
        np.save(
            os.path.join(
                "/home/tejasj/data2/RF_transformer/eval_outputs",
                "unsynchronized",
                f"{id_string}_{testset_identifier}_estimated_bits_{FLAGS.soi_type}"
                f"_{FLAGS.interference_sig_type}",
            ),
            bit1_est,
        )

    if FLAGS.matched_filter_only:
        id_string = "matched_filter"

        sig1_est, bit1_est = run_matched_filter(
            FLAGS.soi_root_dir,
            FLAGS.interference_root_dir,
            FLAGS.soi_type,
            FLAGS.batch_size,
        )
        np.save(
            os.path.join(
                "/home/tejasj/data2/RF_transformer/eval_outputs",
                "unsynchronized",
                f"{id_string}_{testset_identifier}_estimated_soi_{FLAGS.soi_type}"
                f"_{FLAGS.interference_sig_type}",
            ),
            sig1_est,
        )
        np.save(
            os.path.join(
                "/home/tejasj/data2/RF_transformer/eval_outputs",
                "unsynchronized",
                f"{id_string}_{testset_identifier}_estimated_bits_{FLAGS.soi_type}"
                f"_{FLAGS.interference_sig_type}",
            ),
            bit1_est,
        )
    else:
        assert not (FLAGS.decoder_only and FLAGS.wavenet)

        sig1_gt, bit1_gt, sig1_est, bit1_est = run_inference(
            FLAGS.soi_root_dir,
            FLAGS.interference_root_dir,
            FLAGS.checkpoint_dir,
            FLAGS.soi_type,
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            FLAGS.batch_size,
            FLAGS.decoder_only,
            FLAGS.wavenet,
        )

        np.save(
            os.path.join(
                "/home/tejasj/data2/RF_transformer/eval_outputs",
                "unsynchronized",
                f"gt_{testset_identifier}_soi_{FLAGS.soi_type}"
                f"_{FLAGS.interference_sig_type}",
            ),
            sig1_gt,
        )
        np.save(
            os.path.join(
                "/home/tejasj/data2/RF_transformer/eval_outputs",
                "unsynchronized",
                f"gt_{testset_identifier}_bits_{FLAGS.soi_type}"
                f"_{FLAGS.interference_sig_type}",
            ),
            bit1_gt,
        )
        np.save(
            os.path.join(
                "/home/tejasj/data2/RF_transformer/eval_outputs",
                "unsynchronized",
                f"{id_string}_{testset_identifier}_estimated_soi_{FLAGS.soi_type}"
                f"_{FLAGS.interference_sig_type}",
            ),
            sig1_est,
        )
        np.save(
            os.path.join(
                "/home/tejasj/data2/RF_transformer/eval_outputs",
                "unsynchronized",
                f"{id_string}_{testset_identifier}_estimated_bits_{FLAGS.soi_type}"
                f"_{FLAGS.interference_sig_type}",
            ),
            bit1_est,
        )


if __name__ == "__main__":
    app.run(main)
