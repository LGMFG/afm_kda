import time, torch
from fla.layers import KimiDeltaAttention

torch.manual_seed(0)
B, T, D = 1, 4096, 2560 # batch, seq_len, hidden_size
layer = KimiDeltaAttention(hidden_size=D, num_heads=20, head_dim=128,
                           mode="chunk", layer_idx=0).cuda().to(torch.bfloat16)
x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

for _ in range(3):                       # warmup: Triton compiles on first call
    y = layer(x); y = y[0] if isinstance(y, tuple) else y
    y.float().pow(2).mean().backward()
    layer.zero_grad(set_to_none=True); x.grad = None

torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for _ in range(10):
    y = layer(x); y = y[0] if isinstance(y, tuple) else y
    y.float().pow(2).mean().backward()
    layer.zero_grad(set_to_none=True); x.grad = None
torch.cuda.synchronize()
dt = time.time() - t0
print(f"~{10*B*T/dt:,.0f} tokens/s (single layer, fwd+bwd)")
print(f"peak memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")