import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.trainer_config = ml_collections.ConfigDict(
        dict(
            model_dir="",
            distributed=False,
            world_size=1,
            num_workers=0,
            warmup_steps=4000,
            max_steps=1_000_000,
            batch_size=16,
            train_fraction=0.8,
            clip_max_norm=1.0,
            fp16=False,
            log_every=50,
            save_every=5000,
            validate_every=1000,
        )
    )

    config.lr_scheduler_config = [
        "IdentityScheduler",
        ml_collections.ConfigDict(dict()),
    ]

    return config