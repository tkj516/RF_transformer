import os
import pickle

import numpy as np
from absl import app, flags
from tqdm import tqdm

FLAGS = flags.FLAGS
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
flags.DEFINE_string("results_dir", default="", help="Path to the results directory")


BATCH_SIZE = 50
ALL_SINR = np.arange(-30, 0.1, 3)


def run_demod_test(results_dir: str):
    soi_est, bits_est = map(
        lambda sig_type: np.load(
            open(
                os.path.join(
                    results_dir,
                    f"{FLAGS.id_string}_{FLAGS.testset_identifier}_{sig_type}_"
                    f"{FLAGS.soi_type}_{FLAGS.interference_sig_type}.npy",
                ),
                "rb",
            )
        ),
        ["estimated_soi", "estimated_bits"],
    )
   
    soi_gt, bits_gt = map(
        lambda sig_type: np.load(
            open(
                os.path.join(
                    results_dir,
                    f"gt_{FLAGS.testset_identifier}_{sig_type}_"
                    f"{FLAGS.soi_type}_{FLAGS.interference_sig_type}.npy",
                ),
                "rb",
            )
        ),
        ["soi", "bits"],
    )

    # Evaluation pipeline
    def eval_mse(sig_est, sig_soi):
        assert sig_est.shape == sig_soi.shape, f"{sig_est.shape} != {sig_soi.shape}"
        return np.mean(np.abs(sig_est - sig_soi) ** 2, axis=1)

    def eval_ber(bit_est, bit_true):
        assert bit_est.shape == bit_true.shape, "Invalid bit estimate shape"
        return (
            np.sum((bit_est != bit_true).astype(np.float32), axis=1) / bit_true.shape[1]
        )

    all_mse, all_ber = [], []
    for idx, sinr in tqdm(enumerate(ALL_SINR)):
        batch_mse = eval_mse(
            soi_est[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE],
            soi_gt[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE],
        )
        bit_true_batch = bits_gt[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
        batch_ber = eval_ber(
            bits_est[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE], bit_true_batch
        )
        all_mse.append(batch_mse)
        all_ber.append(batch_ber)

    all_mse, all_ber = np.array(all_mse), np.array(all_ber)

    mse_mean = 10 * np.log10(np.mean(all_mse, axis=-1))
    ber_mean = np.mean(all_ber, axis=-1)
    print(f'{"SINR [dB]":>12} {"MSE [dB]":>12} {"BER":>12}')
    print("==================================================")
    for sinr, mse, ber in zip(ALL_SINR, mse_mean, ber_mean):
        print(f"{sinr:>12} {mse:>12,.5f} {ber:>12,.5f}")
    return mse_mean, ber_mean


def main(_):
    soi_type = FLAGS.soi_type
    interference_sig_type = FLAGS.interference_sig_type
    testset_identifier = FLAGS.testset_identifier
    id_string = FLAGS.id_string

    mse_mean, ber_mean = run_demod_test(
        FLAGS.results_dir,
    )

    pickle.dump(
        (mse_mean, ber_mean),
        open(
            os.path.join(
                "eval_outputs/results/unsynchronized",
                f"{id_string}_{testset_identifier}_exports_summary_{soi_type}"
                f"_{interference_sig_type}.pkl",
            ),
            "wb",
        ),
    )

    np.savetxt(
        os.path.join(
            "eval_outputs/results/unsynchronized",
            f"{id_string}_{testset_identifier}_exports_summary_{soi_type}"
            f"_{interference_sig_type}.csv",
        ),
        np.vstack((mse_mean, ber_mean)),
        delimiter=",",
    )


if __name__ == "__main__":
    app.run(main)
