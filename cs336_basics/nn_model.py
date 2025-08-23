import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

from cs336_basics.nn_utils import *

class Linear(nn.Module):
    def __init__(self, d_int, d_out, device=None, dtype=None):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_int, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (d_int + d_out))
        nn.init.normal_(self.weight, mean=0.0, std=std)
        self.weight.data.clamp_(-3*std, 3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model,device=None, dtype=None):
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model, device=device, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=1)
        self.weight.data.clamp_(-3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))   # 可学习缩放参数

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        result = x / rms * self.weight
        return result.to(in_dtype)

class SwigluFFN(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class Rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(Rope, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # Precompute the rotation matrix values (cosine and sine)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)
        angles = positions * inv_freq
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Extract the cosine and sine values for the given token positions
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # Split the input tensor into even and odd dimensions
        x1, x2 = x[..., ::2], x[..., 1::2]

        # Apply the rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Interleave the rotated dimensions back together
        return torch.stack((rotated_x1, rotated_x2), dim=-1).reshape_as(x)
    
def scaled_dot_product_attention(Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.size(-1)
    scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = softmax(scores, dim=-1)
    return einsum(attn, V, "... query key, ... key d_v -> ... query d_v")

class MultiheadSelfAttention(nn.Module):
    def __init__(self, 
        d_model: int,
        num_heads: int,
        max_seq_len: int = 512,
        theta: float = 0,
        device=None, 
        dtype=None
    ): 
        super(MultiheadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope =  Rope(theta, d_model // num_heads, max_seq_len, device=device) if theta > 0 else None
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.size(-2)
        if mask is None:
            mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device))

        q = rearrange(self.q_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        k = rearrange(self.k_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        v = rearrange(self.v_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        attn_output = scaled_dot_product_attention(q, k, v, mask)
        attn_output = rearrange(attn_output, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)", num_heads=self.num_heads)

        return self.output_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 51,
        theta: float = 0,
        device=None, 
        dtype=None
    ):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwigluFFN(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.size(-2)
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        if mask is None:
            mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device))

        x = self.attn(self.ln1(x), token_positions=token_positions, mask=mask) + x
        x = self.ffn(self.ln2(x)) + x
        return x

class TransformerLM(nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None
    ):
        super(TransformerLM, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
