import sys

from regex import F

sys.path.append("..")

import functools
import os
import rfcutils2.qpsk_helper_fn as qpskfn
import rfcutils2.ofdm_helper_fn as ofdmfn
import numpy as np
import tensorflow as tf

from absl.flags import argparse_flags
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_samples", 200000, "Number of samples to generate.")
flags.DEFINE_integer("signal_length", 1280, "Number of bits in the time domain signal.")
flags.DEFINE_string("root_dir", None, "Root directory to store new dataset.")
flags.DEFINE_string("signal_name", None, "Name of the signal.")


def generate_dataset(generation_fn):
    batch_size = 100

    batches = FLAGS.num_samples // batch_size * [batch_size] + [
        FLAGS.num_samples % batch_size
    ]
    all_sig = []
    for b in batches:
        with tf.device("cpu"):
            sig, _, _, _ = generation_fn(b, FLAGS.signal_length)
        sig = sig.numpy()
        all_sig.append(sig)
    sig = np.concatenate(all_sig, axis=0)

    if not os.path.exists(os.path.join("../dataset", FLAGS.root_dir)):
        os.makedirs(os.path.join("../dataset", FLAGS.root_dir), exist_ok=True)

    savedir = os.path.join(
        "../dataset",
        FLAGS.root_dir,
        f"{FLAGS.signal_name}_{FLAGS.num_samples}_{FLAGS.signal_length}",
    )
    if os.path.exists(savedir):
        raise ValueError("Data directory already exists!")
    else:
        os.makedirs(savedir)

    for i in range(sig.shape[0]):
        np.save(os.path.join(savedir, f"sig_{i}.npy"), sig[i, :])

    print(f"Data saved to {savedir}")


def main(_):
    parser = argparse_flags.ArgumentParser(
        description="Arguments for generating RF datasets."
    )
    subparsers = parser.add_subparsers(help="Subparsers for different data types")

    parser_qpsk = subparsers.add_parser("qpsk", help="Generate QPSK dataset.")
    parser_qpsk.set_defaults(
        func=functools.partial(
            generate_dataset, generation_fn=qpskfn.generate_qpsk_signal
        )
    )

    parser_ofdm = subparsers.add_parser("ofdm", help="Generate general OFDM dataset.")
    parser_ofdm.set_defaults(
        func=functools.partial(
            generate_dataset, generation_fn=ofdmfn.generate_ofdm_signal
        )
    )

    args = parser.parse_args()
    args.func()


if __name__ == "__main__":
    app.run(main)
