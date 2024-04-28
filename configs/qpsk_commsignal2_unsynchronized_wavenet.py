import configs.base_configs as base_configs
import ml_collections


def get_config():
    config = base_configs.get_config()

    config.dataset_config = [
        "UnsynchronizedRFDatasetWaveNet",
        dict(
            soi_root_dir="/home/tejasj/data2/RF_transformer/dataset/qpsk/qpsk_100000_2561",
            interference_root_dir="/home/tejasj/data2/RF_transformer/dataset/CommSignal2",
            signal_length=40960,
            number_soi_offsets=16,
            use_rand_phase=True,
        ),
    ]

    config.model_config = [
        "WaveNet",
        ml_collections.ConfigDict(
            dict(
                input_channels=2,
                residual_channels=128,
                dilation_cycle_length=10,
                residual_layers=30,
            )
        ),
    ]

    config.optimizer_config = [
        "AdamW",
        ml_collections.ConfigDict(
            dict(
                lr=5e-5,
                weight_decay=0.01,
            )
        ),
    ]

    config.lr_scheduler_config = [
        "ReduceLROnPlateau",
        ml_collections.ConfigDict(dict()),
    ]

    config.trainer_config.model_dir = (
        "checkpoints/qpsk_commsignal2_unsynchronized_wavenet"
    )
    config.trainer_config.batch_size = 16
    config.trainer_config.train_fraction = 0.95
    config.trainer_config.distributed = True
    config.trainer_config.world_size = 2
    config.trainer_config.clip_max_norm = -1

    return config
