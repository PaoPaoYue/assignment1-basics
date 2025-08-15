import torch
import numpy as np

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