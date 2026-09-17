# This script tests the KimiDeltaAttention layer with a segment-id mask. It runs two forward passes with different inputs and compares the outputs to check for consistency.

import torch
from fla.layers.kda import KimiDeltaAttention

torch.manual_seed(0)
L = KimiDeltaAttention(hidden_size=2560, num_heads=20, head_dim=128,
                       conv_size=4, mode="chunk", layer_idx=0).cuda().to(torch.bfloat16)

LA, LB = 301, 200
A  = torch.randn(1, LA, 2560, dtype=torch.bfloat16).cuda()
A2 = torch.randn(1, LA, 2560, dtype=torch.bfloat16).cuda()
B  = torch.randn(1, LB, 2560, dtype=torch.bfloat16).cuda()
seg = torch.cat([torch.ones(1, LA), 2 * torch.ones(1, LB)], 1).long().cuda()

@torch.no_grad()
def run(pre):
    o, _, _ = L(hidden_states=torch.cat([pre, B], 1), attention_mask=seg, use_cache=False)
    return o[0, LA:].float()

print("segment-id mask straight to KDA:", f"{(run(A) - run(A2)).abs().max().item():.3e}")