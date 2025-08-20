from dataclasses import dataclass
import torch
import os
from typing import Any, BinaryIO, IO

def save_checkpoint(
    out: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler = None,
    **kwargs
):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    checkpoint_data = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict() if optimizer else None,
        scheduler_state_dict=scheduler.state_dict() if scheduler else None,
        model_info=kwargs
    )
    torch.save(checkpoint_data, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkpoint_data = torch.load(src, weights_only=False)
    return (checkpoint_data["model_state_dict"], 
            checkpoint_data["optimizer_state_dict"], 
            checkpoint_data["scheduler_state_dict"], 
            checkpoint_data["model_info"])
