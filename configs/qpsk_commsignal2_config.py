import configs.base_configs as base_configs
import ml_collections


def get_config():
    config = base_configs.get_config()

    config.dataset_config = [
        "ICASSPDataset",
        dict(
            root_dir="/home/tejasj/data2/RF_transformer/npydataset/Dataset_QPSK_CommSignal2_Mixture",
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
                block_size=320,
                causal_encoder=False,
                causal_decoder=True,
                max_seq_len=320,
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

    config.trainer_config.model_dir = "checkpoints/qpsk_commsignal2"
    config.trainer_config.batch_size = 32
    config.trainer_config.train_fraction=0.90

    return config
