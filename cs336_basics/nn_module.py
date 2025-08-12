import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

class Linear(nn.Module):
    def __init__(self, d_int, d_out, weights=None, device=None, dtype=None):
        super(Linear, self).__init__()
        if weights is not None:
            assert weights.shape == (d_out, d_int), "weights shape mismatch"
            self.weights = nn.Parameter(weights)
            return
        self.weights = nn.Parameter(torch.randn(d_out, d_int, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (d_int + d_out))
        torch.init.normal_(self.weights, mean=0.0, std=std)
        self.weights.data.clamp_(-3*std, 3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
    
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, weights=None, device=None, dtype=None):
        super(Embedding, self).__init__()
        if weights is not None:
            assert weights.shape == (vocab_size, d_model), "weights shape mismatch"
            self.weights = nn.Parameter(weights)
            return
        self.weights = nn.Parameter(torch.randn(vocab_size, d_model, device=device, dtype=dtype))
        torch.init.normal_(self.weights, mean=0.0, std=1)
        self.weights.data.clamp_(-3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights[x]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8, weights=None, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        if weights is not None:
            assert weights.shape == (d_model,), "weights shape mismatch"
            self.weights = nn.Parameter(weights)
            return
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))   # 可学习缩放参数

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        result = x / rms * self.weights
        return result.to(in_dtype)

class SwigluFFN(nn.Module):
    def __init__(self, d_model, d_ff, w1_weights=None, w2_weights=None, w3_weights=None, device=None, dtype=None):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, weights=w1_weights, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, weights=w2_weights, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, weights=w3_weights, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x)) * self.linear3(x))