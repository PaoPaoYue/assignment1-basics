import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable
from einops import einsum, rearrange

def get_model_size(model: nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def softmax(in_features:torch.Tensor, dim: int) -> torch.Tensor:
    exp_values = torch.exp(in_features - in_features.max(dim=dim, keepdim=True).values)
    return exp_values / exp_values.sum(dim=dim, keepdim=True)

def cross_entropy(inputs:torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(inputs, dim=-1)
    nll = -torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean(dim=-1)

def perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.exp(nll.mean(dim=-1))

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-6) -> float:
    grad_norm = []
    for p in parameters:
        if p.grad is not None:
            grad_norm.append(p.grad.data.norm(2))
    grad_norm = torch.norm(torch.stack(grad_norm), 2)
    if max_l2_norm > 0 and grad_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(max_l2_norm / (grad_norm + eps))
    return grad_norm.item()

def top_p_sampling(inputs:torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(inputs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs >= top_p
    if torch.any(mask):
        first_mask_idx = torch.argmax(mask.int(), dim=-1)
        sorted_indices = [sorted_indices[i, first_mask_idx[i]+1:] for i in range(mask.shape[0])]
        outputs = inputs.clone()
        for i in range(mask.shape[0]):
            outputs[i, sorted_indices[i]] = 0
        outputs = outputs / outputs.sum(dim=-1, keepdim=True)
    return outputs