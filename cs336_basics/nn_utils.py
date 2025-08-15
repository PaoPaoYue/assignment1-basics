import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable
from einops import einsum, rearrange

def softmax(in_features:torch.Tensor, dim: int) -> torch.Tensor:
    exp_values = torch.exp(in_features - in_features.max(dim=dim, keepdim=True).values)
    return exp_values / exp_values.sum(dim=dim, keepdim=True)

def cross_entropy(inputs:torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(inputs, dim=-1)
    batch_entropy = -torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    return batch_entropy.mean(dim=-1)

def perplexity(inputs:torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    #inputs [B, T, D] targets [B, T]
    with torch.no_grad():
        log_probs = F.log_softmax(inputs, dim=-1)
        batch_perplexity = -torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
        return batch_perplexity.mean(dim=-1)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-6):
    grad_norm = []
    for p in parameters:
        if p.grad is not None:
            grad_norm.append(p.grad.data.norm(2))
    grad_norm = torch.norm(torch.stack(grad_norm), 2)
    if grad_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(max_l2_norm / (grad_norm + eps))