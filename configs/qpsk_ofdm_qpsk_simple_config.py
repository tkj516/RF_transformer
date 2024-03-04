import configs.base_configs as base_configs
import ml_collections


def get_config():
    config = base_configs.get_config()

    config.dataset_config = [
        "RFDatasetBase",
        dict(
            soi_root_dir=(
                "/home/tejasj/data2/RF_transformer/dataset/qpsk/qpsk_200000_160"
            ),
            interference_root_dir=(
                "/home/tejasj/data2/RF_transformer/dataset/ofdm/ofdm_200000_32"
            ),
            window_size=128,
            context_size=32,
        ),
    ]

    config.model_config = [
        "Transformer",
        ml_collections.ConfigDict(
            dict(
                input_dim=320,
                output_dim=256,
                n_encoder_layers=12,
                n_decoder_layers=12,
                embed_dim=768,
                n_head=12,
                bias=False,
                dropout=0.0,
                block_size=30,
                causal_encoder=False,
                causal_decoder=True,
                max_seq_len=30,
            )
        ),
    ]

    config.optimizer_config = [
        "AdamW",
        ml_collections.ConfigDict(
            dict(
                lr=0.0006,
                weight_decay=0.01,
            )
        ),
    ]

    config.trainer_config.model_dir = "checkpoints/qpsk_ofdm_qpsk_simple"
    config.trainer_config.batch_size = 16
    config.trainer_config.train_fraction=0.90

    return config
