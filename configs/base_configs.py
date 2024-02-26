import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.image_resolution=64
    config.n_samples=16

    config.trainer_config = ml_collections.ConfigDict(
        dict(
            model_dir="",
            distributed=False,
            world_size=1,
            num_workers=0,
            log_every=50,
            save_every=5000,
            validate_every=1000,
            spectrum_every=1000,
            max_steps=1_000_000,
            batch_size=1,
            train_fraction=0.8,
            clip_max_norm=1e9,
            fp16=False,
            joint=False,
            nesting=True,
            spectral_norm_lambda=0.0,
            norm="l1",
        )
    )

    return config