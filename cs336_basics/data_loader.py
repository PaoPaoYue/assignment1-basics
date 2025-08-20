import torch
import numpy as np

from torch.utils.data import Dataset, Sampler
import numpy.typing as npt


import torch
from torch.utils.data import Sampler
import numpy as np

class TokensDataset(Dataset):
    def __init__(self, data: npt.ArrayLike, context_length: int):
        self.data = data
        self.context_length = context_length
    def __len__(self):
        return self.data.shape[0] - self.context_length
    def __getitem__(self, idx):
        return (
            torch.from_numpy(
                self.data[idx:idx + self.context_length]
            ).long(),
            torch.from_numpy(
                self.data[idx + 1:idx + self.context_length + 1]
            ).long()
        )


class RandomStartBatchSampler(Sampler[list[int]]):
    def __init__(self, max_start: int, batch_size: int, drop_last: bool = False):
        self.max_start = max_start
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        if self.drop_last:
            num_batches = self.max_start // self.batch_size
        else:
            num_batches = (self.max_start + self.batch_size - 1) // self.batch_size

        for _ in range(num_batches):
            yield np.random.randint(0, self.max_start, size=self.batch_size)

    def __len__(self):
        if self.drop_last:
            return self.max_start // self.batch_size
        return (self.max_start + self.batch_size - 1) // self.batch_size

def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get a batch of data from the dataset.
    
    Args:
        dataset (np.ndarray): The dataset to sample from.
        batch_size (int): The number of samples in the batch.
        context_length (int): The length of the context for each sample.
        device (str): The device to which the tensors should be moved.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
    """
    indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = torch.from_numpy(np.array([dataset[i:i + context_length] for i in indices], dtype=np.int64)).to(device)
    y = torch.from_numpy(np.array([dataset[i + 1:i + context_length + 1] for i in indices], dtype=np.int64)).to(device)
    return x, y