import os
import h5py

import numpy as np

from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "dataset_path", default="", help="Path containing the dataset"
)
flags.DEFINE_string(
    "interference_sig_type", default="CommSignal2", help="Type of interference signal"
)
flags.DEFINE_string(
    "root_dir",
    default="/home/tejasj/data2/RF_transformer/dataset",
    help="Root directory for the dataset",
)


def hypy_to_npy_dataset(h5py_file: str, root_dir: str):
    with h5py.File(h5py_file, "r") as data_h5file:
        sig_data = np.array(data_h5file.get("dataset"))
        sig_type_info = data_h5file.get("sig_type")[()]
        if isinstance(sig_type_info, bytes):
            sig_type_info = sig_type_info.decode("utf-8")

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        for idx, sig in enumerate(sig_data):
            np.save(
                os.path.join(root_dir, f"sig_{idx}.npy"),
                sig,
            )


def main(_):
    root_dir = os.path.join(FLAGS.root_dir, FLAGS.interference_sig_type)

    h5py_file = os.path.join(FLAGS.dataset_path)

    hypy_to_npy_dataset(h5py_file, root_dir)


if __name__ == "__main__":
    app.run(main)
