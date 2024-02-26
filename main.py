import importlib
import socketserver

from absl import app, flags
from ml_collections import ConfigDict, config_flags
from torch.cuda import device_count
from torch.multiprocessing import spawn

CONFIG = config_flags.DEFINE_config_file("config")
FLAGS = flags.FLAGS
flags.DEFINE_string("trainer", default="low_rank_dsm", help="The trainer type.")


def _get_free_port():
    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]


def main(_):
    cfg = ConfigDict(CONFIG.value)
    trainer = importlib.__import__(f"train_{FLAGS.trainer}")

    print("##########################################")
    print(f"Training with train_{FLAGS.trainer}.")
    print("##########################################")

    # Setup training
    world_size = device_count()
    if cfg.trainer_config.distributed and world_size != cfg.trainer_config.world_size:
        raise ValueError(
            "Requested world size is not the same as number of visible GPUs."
        )
    if cfg.trainer_config.distributed:
        if world_size < 2:
            raise ValueError(
                "Distributed training cannot be run on machine "
                f"with {world_size} device(s)."
            )
        if cfg.trainer_config.batch_size % world_size != 0:
            raise ValueError(
                f"Batch size {cfg.trainer_config.batch_size} is not evenly "
                f"divisble by # GPUs = {world_size}."
            )
        cfg.trainer_config.batch_size = cfg.trainer_config.batch_size // world_size
        port = _get_free_port()
        spawn(
            trainer.train_distributed,
            args=(world_size, port, cfg),
            nprocs=world_size,
            join=True,
        )
    else:
        trainer.train(cfg)


if __name__ == "__main__":
    app.run(main)