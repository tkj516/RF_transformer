import configs.base_configs as base_configs
import ml_collections


def get_config():
    config = base_configs.get_config()

    config.dataset_config = [
        "UnsynchronizedRFDataset",
        dict(
            soi_root_dir="../dataset/qpsk/qpsk_100000_2561",
            interference_root_dir="../dataset/CommSignal5G1",
            window_size=128,
            context_size=(32, 32),
            signal_length=40960,
            number_soi_offsets=16,
            use_rand_phase=True,
        ),
    ]

    config.model_config = [
        "Transformer",
        ml_collections.ConfigDict(
            dict(
                input_dim=384,
                output_dim=256,
                n_decoder_layers=14,
                embed_dim=768,
                n_head=12,
                bias=False,
                dropout=0.0,
                block_size=384,
                causal_decoder=True,
                max_seq_len=384,
            )
        ),
    ]

    config.optimizer_config = [
        "AdamW",
        ml_collections.ConfigDict(
            dict(
                lr=0.0001,
                weight_decay=0.01,
            )
        ),
    ]

    config.lr_scheduler_config = [
        "ReduceLROnPlateau",
        ml_collections.ConfigDict(dict()),
    ]

    config.trainer_config.model_dir="checkpoints/qpsk_commsignal5g1_unsynchronized_decoder"
    config.trainer_config.batch_size=40
    config.trainer_config.train_fraction=0.95
    config.trainer_config.distributed=True
    config.trainer_config.world_size=2

    return config
