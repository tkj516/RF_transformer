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

tf.config.set_visible_devices([], "GPU")


FLAGS = flags.FLAGS
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
    inputs: np.ndarray, window_size: int, context_size: int
) -> torch.Tensor:
    inputs = torch.view_as_real(torch.from_numpy(inputs)).to(torch.float32)
    inputs = F.pad(inputs, (0, 0, context_size, 0, 0, 0))
    input_windows = inputs.unfold(1, context_size + window_size, window_size).reshape(
        inputs.shape[0], -1, (window_size + context_size) * 2
    )
    return input_windows


@torch.no_grad()
def run_inference(
    checkpoint_dir: str,
    mixtures: np.ndarray,
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
    model.load_state_dict(checkpoint["model"])
    model.eval()

    window_size = config.dataset_config[1]["window_size"]
    context_size = config.dataset_config[1]["context_size"]

    mixtures = process_inputs(
        mixtures,
        window_size,
        context_size,
    )

    all_soi_est = []
    for mixture in tqdm(torch.split(mixtures, batch_size, dim=0)):
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
        waveform = torch.view_as_complex(waveform).numpy()
        all_soi_est.append(waveform)
    soi_est = np.concatenate(all_soi_est, axis=0)

    bit_est = []
    for est in np.split(soi_est, batch_size):
        bit_est_batch, _ = demod_soi(est)
        bit_est.append(bit_est_batch.numpy())
    bit_est = np.concatenate(bit_est, axis=0)
    return soi_est, bit_est


def main(_):
    soi_type = FLAGS.soi_type
    interference_sig_type = FLAGS.interference_sig_type
    testset_identifier = FLAGS.testset_identifier
    id_string = FLAGS.id_string

    all_sig_mixture = np.load(
        os.path.join(
            "/home/tejasj/data2/dataset",
            f"{testset_identifier}_testmixture_{soi_type}_{interference_sig_type}.npy",
        )
    )
    sig1_est, bit1_est = run_inference(
        FLAGS.checkpoint_dir,
        all_sig_mixture,
        soi_type,
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        FLAGS.batch_size,
    )

    if not os.path.exists("eval_outputs"):
        os.makedirs("eval_outputs")

    np.save(
        os.path.join(
            "eval_outputs",
            f"{id_string}_{testset_identifier}_estimated_soi_{soi_type}"
            f"_{interference_sig_type}",
        ),
        sig1_est,
    )
    np.save(
        os.path.join(
            "eval_outputs",
            f"{id_string}_{testset_identifier}_estimated_bits_{soi_type}"
            f"_{interference_sig_type}",
        ),
        bit1_est,
    )


if __name__ == "__main__":
    app.run(main)
