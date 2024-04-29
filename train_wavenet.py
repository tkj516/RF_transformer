"""
WaveNet for RF Source Separation.
"""

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.wavenet import Wave
import rfcutils2
from utils.class_builder import ClassBuilder
from utils.dictionary_to_string import dict_to_str
from utils.lr_schedulers import CosineAnnealingWarmUp, IdenityScheduler
from utils.rf_dataset import UnsynchronizedRFDatasetWaveNet
from utils.utils import (
    MyDistributedDataParallel,
    get_train_val_dataset,
    nested_to_device,
)

torch.backends.cudnn.benchmark = True


MODELS_REGISTER = {
    "WaveNet": Wave,
}
image_transforms_builder = ClassBuilder(MODELS_REGISTER)


OPTIMIZER_REGISTER = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
}
optimizer_builder = ClassBuilder(OPTIMIZER_REGISTER)


LR_SCHEDULER_REGISTER = {
    "IdentityScheduler": IdenityScheduler,
    "CosineAnnealingWarmUp": CosineAnnealingWarmUp,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}
lr_scheduler_builder = ClassBuilder(LR_SCHEDULER_REGISTER)


DATASET_REGISTER = {
    "UnsynchronizedRFDatasetWaveNet": UnsynchronizedRFDatasetWaveNet,
}
dataset_register = ClassBuilder(DATASET_REGISTER)


class Learner:
    def __init__(self, model: nn.Module, cfg: ConfigDict, rank: int):
        # Store some important variables
        self.rank = rank
        self.cfg = cfg
        self.step = 0

        # Instantiate the dataloaders
        self.build_dataloaders()

        # Store the model
        self.model = model

        # Build the optimizer
        self.build_optimizer()

        # Instantiate the leanring rate scheduler
        self.lr_scheduler, _ = lr_scheduler_builder.build(
            cfg.lr_scheduler_config, optimizer=self.optimizer
        )
        self.autocast = torch.cuda.amp.autocast(enabled=cfg.trainer_config.fp16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.trainer_config.fp16)

        # Define the loss functions
        self.loss_fn = F.mse_loss

        # Instantiate a Tensorboard summary writer
        self.writer = SummaryWriter(cfg.trainer_config.model_dir)
        # Write the config to Tensorboard
        self.writer.add_text("config", dict_to_str(self.cfg.to_dict()))

    @property
    def is_master(self):
        return self.rank == 0

    def build_optimizer(self):
        # Create the optimizer
        self.optimizer, _ = optimizer_builder.build(
            self.cfg.optimizer_config,
            params=self.model.parameters(),
            fused=True,
        )

    def build_dataloaders(self):
        self.dataset, _ = dataset_register.build(
            self.cfg.dataset_config,
        )
        self.train_dataset, self.val_dataset = get_train_val_dataset(
            self.dataset, self.cfg.trainer_config.train_fraction
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.trainer_config.batch_size,
            shuffle=not self.cfg.trainer_config.distributed,
            num_workers=(
                self.cfg.trainer_config.num_workers
                if self.cfg.trainer_config.distributed
                else 0
            ),
            sampler=DistributedSampler(
                self.train_dataset,
                num_replicas=self.cfg.trainer_config.world_size,
                rank=self.rank,
            )
            if self.cfg.trainer_config.distributed
            else None,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.trainer_config.batch_size,
            shuffle=not self.cfg.trainer_config.distributed,
            num_workers=(
                self.cfg.trainer_config.num_workers
                if self.cfg.trainer_config.distributed
                else 0
            ),
            sampler=DistributedSampler(
                self.val_dataset,
                num_replicas=self.cfg.trainer_config.world_size,
                rank=self.rank,
            )
            if self.cfg.trainer_config.distributed
            else None,
            pin_memory=True,
        )

    def state_dict(self):
        return {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "cfg": self.cfg.to_dict(),
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        self.step = state_dict["step"]

    def save_to_checkpoint(self, filename="weights"):
        save_basename = f"{filename}-{self.step}.pt"
        save_name = f"{self.cfg.trainer_config.model_dir}/{save_basename}"
        link_name = f"{self.cfg.trainer_config.model_dir}/{filename}.pt"
        torch.save(self.state_dict(), save_name)

        if os.path.islink(link_name):
            os.unlink(link_name)
        os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename="weights"):
        try:
            checkpoint = torch.load(
                f"{self.cfg.trainer_config.model_dir}/{filename}.pt"
            )
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self):
        device = next(self.model.parameters()).device
        while True:
            iterable = self.train_dataloader
            if self.rank == 0:
                iterable = tqdm(
                    iterable,
                    desc=(
                        f"Training ({self.step}"
                        f" / {self.cfg.trainer_config.max_steps})"
                    ),
                )
            for _, inputs in enumerate(iterable):
                # TODO: Add option for gradient accumulation
                inputs = nested_to_device(inputs, device)
                loss = self.train_step(inputs, logging_rank=self.rank == 0)

                # Check for NaNs
                if torch.isnan(loss).any():
                    raise RuntimeError(f"Detected NaN loss at step {self.step}.")

                if self.is_master:
                    if self.step % self.cfg.trainer_config.save_every == 0:
                        self.save_to_checkpoint()

                if (
                    self.step >= 0
                    and self.step % self.cfg.trainer_config.validate_every == 0
                ):
                    loss = self.validate()

                if self.cfg.trainer_config.distributed:
                    dist.barrier()

                self.step += 1

                if self.step == self.cfg.trainer_config.max_steps:
                    if self.is_master and self.cfg.trainer_config.distributed:
                        self.save_to_checkpoint()
                        print("Ending training...")
                    dist.barrier()
                    exit(0)

            loss = self.validate()
            self.lr_scheduler.step(loss)

    def train_step(
        self,
        inputs: List[torch.Tensor],
        logging_rank: bool = False,
    ) -> torch.Tensor:
        self.optimizer.zero_grad()

        mixture = inputs["mixture"]
        target = inputs["soi"]

        with self.autocast:
            preds = self.model(mixture)
            loss = self.loss_fn(preds, target)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if self.cfg.trainer_config.clip_max_norm > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.trainer_config.clip_max_norm
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if logging_rank and self.step % self.cfg.trainer_config.log_every == 0:
            self.writer.add_scalar("train/loss", loss, self.step)

        return loss

    @torch.no_grad()
    def validate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        self.model.eval()

        loss = 0
        all_soi_est = []
        all_soi_gt = []

        iterable = self.val_dataloader
        if self.rank == 0:
            iterable = tqdm(
                iterable,
                desc=f"Running validation after step {self.step}",
            )
        for inputs in iterable:
            with self.autocast:
                inputs = nested_to_device(inputs, device)

                mixture = inputs["mixture"]
                target = inputs["soi"]

                preds = self.model(mixture)
                loss += (
                    self.loss_fn(preds, target)
                    * mixture.shape[0]
                    / len(self.val_dataset)
                )

            # Recover the waveforms in the time domain
            pred_waveform = torch.view_as_complex(
                preds.permute(0, 2, 1).contiguous()
            ).cpu()
            target_waveform = torch.view_as_complex(
                target.permute(0, 2, 1).contiguous()
            ).cpu()

            if isinstance(self.dataset, UnsynchronizedRFDatasetWaveNet):
                slice_idx = inputs["soi_offset"].reshape(-1, 1) + torch.arange(
                    self.dataset.signal_length - self.dataset.number_soi_offsets,
                    device=inputs["soi_offset"].device,
                ).reshape(1, -1)
                pred_waveform = torch.take_along_dim(
                    pred_waveform, slice_idx.to(pred_waveform.device), dim=1
                )
                target_waveform = torch.take_along_dim(
                    target_waveform, slice_idx.to(target_waveform.device), dim=1
                )

            all_soi_est.append(pred_waveform.numpy())
            all_soi_gt.append(target_waveform.numpy())

        # Compute the bit error rate (BER)
        soi_est = np.concatenate(all_soi_est, axis=0)
        soi_gt = np.concatenate(all_soi_gt, axis=0)

        demod_soi = rfcutils2.qpsk_matched_filter_demod

        bit_est = []
        bit_gt = []
        for est, gt in zip(np.split(soi_est, 25), np.split(soi_gt, 25)):
            bit_est_batch, _ = demod_soi(est)
            bit_gt_batch, _ = demod_soi(gt)
            bit_est.append(bit_est_batch.numpy())
            bit_gt.append(bit_gt_batch.numpy())
        bit_est = np.concatenate(bit_est, axis=0)
        bit_gt = np.concatenate(bit_gt, axis=0)

        ber = torch.tensor(
            np.mean(bit_est != bit_gt) * bit_est.shape[0] / len(self.val_dataset),
            device=loss.device,
        )

        if self.cfg.trainer_config.distributed:
            dist.reduce(loss, 0, op=dist.ReduceOp.SUM)
            dist.reduce(ber, 0, op=dist.ReduceOp.SUM)

        if self.rank == 0:
            self.writer.add_scalar("val/loss", loss, self.step)
            self.writer.add_scalar("val/ber", ber, self.step)

        self.model.train()

        return loss


def _train_impl(rank: int, model: nn.Module, cfg: ConfigDict):
    torch.backends.cudnn.benchmark = True

    learner = Learner(model, cfg, rank)
    # learner.restore_from_checkpoint()
    learner.train()


def train(cfg: ConfigDict):
    """Training on a single GPU."""
    model, _ = image_transforms_builder.build(cfg.model_config)
    model.cuda()
    _train_impl(0, model, cfg)


def init_distributed(rank: int, world_size: int, port: str):
    """Initialize distributed training on multiple GPUs."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def train_distributed(rank: int, world_size: int, port, cfg: ConfigDict):
    """Training on multiple GPUs."""
    init_distributed(rank, world_size, port)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    model, _ = image_transforms_builder.build(cfg.model_config)
    model.to(device)
    model = MyDistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True
    )
    _train_impl(rank, model, cfg)
