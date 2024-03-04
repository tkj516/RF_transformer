import math
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWarmUp(LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=6e-5, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(CosineAnnealingWarmUp, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [
                base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs
            ]
        elif self.last_epoch > self.T_max:
            return [self.eta_min for _ in self.base_lrs]
        else:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.last_epoch - self.T_warmup)
                        / (self.T_max - self.T_warmup)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]
