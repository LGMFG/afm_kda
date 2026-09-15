"""A6: KDA vs full attention as sequence length grows -- the headline figure.

Attention is O(T^2): its tokens/s should roughly halve each time T doubles.
KDA is O(T): its tokens/s should stay flat.  Same hidden size, same dtype,
same device, forward+backward, so the only variable is the mixing rule.
"""
import os, sys, gc, time, torch
import torch.nn as nn
import torch.nn.functional as F
WORK = os.environ["WORK_LUKAS"]
sys.path.insert(0, WORK)

from fla.layers.kda import KimiDeltaAttention

H, N_HEADS, N_KV, HD = 2560, 20, 4, 128      # AFM's real geometry
DTYPE, DEV = torch.bfloat16, "cuda"
LENGTHS = [2048, 4096, 8192, 16384, 32768]


class Attention(nn.Module):
    """AFM-shaped attention: GQA projections + causal SDPA + output projection."""
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(H, N_HEADS * HD, bias=False)
        self.k = nn.Linear(H, N_KV * HD, bias=False)
        self.v = nn.Linear(H, N_KV * HD, bias=False)
        self.o = nn.Linear(N_HEADS * HD, H, bias=False)
        self.n_rep = N_HEADS // N_KV

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, N_HEADS, HD).transpose(1, 2)
        k = self.k(x).view(b, t, N_KV, HD).transpose(1, 2)
        v = self.v(x).view(b, t, N_KV, HD).transpose(1, 2)
        k = k.repeat_interleave(self.n_rep, dim=1)      # group-major GQA expansion
        v = v.repeat_interleave(self.n_rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(o.transpose(1, 2).reshape(b, t, -1))


def bench(layer, T, warmup=2, iters=5):
    """tokens/s and peak memory for one fwd+bwd at sequence length T."""
    x = torch.randn(1, T, H, device=DEV, dtype=DTYPE, requires_grad=True)
    for _ in range(warmup):
        out = layer(x)
        (out[0] if isinstance(out, tuple) else out).float().pow(2).mean().backward()
        layer.zero_grad(set_to_none=True); x.grad = None
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = layer(x)
        (out[0] if isinstance(out, tuple) else out).float().pow(2).mean().backward()
        layer.zero_grad(set_to_none=True); x.grad = None
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return T / dt, torch.cuda.max_memory_allocated() / 1e9


print(f"hidden {H}, {N_HEADS} heads x {HD}, bf16, batch 1, fwd+bwd\n")
print(f"{'T':>7} | {'KDA tok/s':>12} {'KDA GB':>8} | {'attn tok/s':>12} {'attn GB':>8} | {'speedup':>8}")
print("-" * 76)
for T in LENGTHS:
    kda = KimiDeltaAttention(hidden_size=H, num_heads=N_HEADS, head_dim=HD,
                             conv_size=4, mode="chunk", layer_idx=0).to(DEV, DTYPE)
    k_tps, k_mem = bench(kda, T)
    del kda; gc.collect(); torch.cuda.empty_cache()

    attn = Attention().to(DEV, DTYPE)
    a_tps, a_mem = bench(attn, T)
    del attn; gc.collect(); torch.cuda.empty_cache()

    print(f"{T:>7} | {k_tps:>12,.0f} {k_mem:>8.2f} | {a_tps:>12,.0f} {a_mem:>8.2f} | {k_tps/a_tps:>7.2f}x")