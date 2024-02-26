import torch

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset
from typing import List, Tuple


class MyDistributedDataParallel(DistributedDataParallel):
    def __init__(self, model, **kwargs):
        super(MyDistributedDataParallel, self).__init__(model, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        

def nested_to_device(
        inputs: List[torch.Tensor], 
        device: torch.device
) -> List[torch.Tensor]:
    """Push all tensors in a list to the specified device."""
    for i, t in enumerate(inputs):
        inputs[i] = t.to(device)
    return inputs


def get_train_val_dataset(
        dataset: Dataset,
        train_fraction: float
) -> Tuple[Dataset, Dataset]:
    """Splits a dataset into trainining and validation sets."""
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_fraction, 1 - train_fraction],
        generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset


def extract(
    vals: torch.Tensor,
    index: torch.Tensor,
    shape: List[int],
) -> torch.Tensor:
    batch_size = index.shape[0]
    out = vals.gather(-1, index.cpu())
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(index.device)