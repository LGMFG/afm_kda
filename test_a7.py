"""A7: does the recurrent state reset at packed document boundaries?

A recurrent layer sweeps left to right carrying state.  Packing puts several
documents in one row, so without cu_seqlens document B inherits document A's
state -- silently, with a normal-looking loss.

Pure KDA (all 36 layers) so any contamination is KDA's, not attention's.
Checkpointing on and off, because that is the path the reference loses it on.
"""
import os, sys, torch
WORK = os.environ["WORK_LUKAS"]
sys.path.insert(0, WORK)
torch.manual_seed(0)

from transformers import AutoTokenizer
from afm_kda.configuration_afm_kda import AFMKDAConfig
from afm_kda.modeling_afm_kda import (
    AFMKDAForCausalLM, init_new_parameters, transfer_attention_into_kda,
)
from transformers import AutoModelForCausalLM

TEACHER = "arcee-ai/AFM-4.5B-Base"
DTYPE, DEV = torch.bfloat16, "cuda"
LEN_A, LEN_B = 301, 200          # deliberately NOT multiples of the chunk size (64)

teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=DTYPE).to(DEV).eval()
cfg = AFMKDAConfig.from_pretrained(TEACHER, kda_layers=list(range(36)),
                                   use_rope=False, use_attn_gate=False)
model, info = AFMKDAForCausalLM.from_pretrained(
    TEACHER, config=cfg, dtype=DTYPE, attn_implementation="sdpa",
    output_loading_info=True)
init_new_parameters(model, info.get("missing_keys", []))
transfer_attention_into_kda(model, teacher, verbose=False)
model = model.to(DEV).eval()
del teacher; torch.cuda.empty_cache()

tok = AutoTokenizer.from_pretrained(TEACHER)
vocab = cfg.vocab_size
ids_A = torch.randint(0, vocab, (1, LEN_A), device=DEV)
ids_B = torch.randint(0, vocab, (1, LEN_B), device=DEV)
packed = torch.cat([ids_A, ids_B], dim=1)
cu = torch.tensor([0, LEN_A, LEN_A + LEN_B], dtype=torch.int32, device=DEV)


def b_logits(checkpointing):
    """Document B's logits, three ways, with checkpointing on or off."""
    if checkpointing:
        model.gradient_checkpointing_enable()
        model.train()                     # checkpointing only engages in train mode
        ctx = torch.enable_grad()
    else:
        model.gradient_checkpointing_disable()
        model.eval()
        ctx = torch.no_grad()
    with ctx:
        alone  = model(ids_B).logits
        with_cu = model(packed, cu_seqlens=cu).logits[:, LEN_A:, :]
        no_cu   = model(packed).logits[:, LEN_A:, :]
    return alone.detach().float(), with_cu.detach().float(), no_cu.detach().float()


def rel(a, b):
    return ((a - b).abs().max() / a.abs().max()).item()


for ckpt in (False, True):
    alone, with_cu, no_cu = b_logits(ckpt)
    print(f"\n=== gradient_checkpointing = {ckpt} ===")
    print(f"  packed WITH cu_seqlens vs B alone : {rel(alone, with_cu):.3e}   <- must be SMALL")
    print(f"  packed WITHOUT cu_seqlens vs alone: {rel(alone, no_cu):.3e}   <- must be LARGE")
    print(f"  PASS: {rel(alone, with_cu) < 1e-2 and rel(alone, no_cu) > 1e-1}")



ids_A2 = torch.randint(0, vocab, (1, LEN_A), device=DEV)   # different content, SAME length
packed2 = torch.cat([ids_A2, ids_B], dim=1)

with torch.no_grad():
    model.gradient_checkpointing_disable(); model.eval()
    b_from_A  = model(packed,  cu_seqlens=cu).logits[:, LEN_A:, :].float()
    b_from_A2 = model(packed2, cu_seqlens=cu).logits[:, LEN_A:, :].float()
    b_nocu_A  = model(packed ).logits[:, LEN_A:, :].float()
    b_nocu_A2 = model(packed2).logits[:, LEN_A:, :].float()

print("\n=== decisive test: does B depend on what precedes it? ===")
print(f"  WITH cu_seqlens,    A vs A': {rel(b_from_A, b_from_A2):.3e}   <- must be ~0")
print(f"  WITHOUT cu_seqlens, A vs A': {rel(b_nocu_A, b_nocu_A2):.3e}   <- must be LARGE")

d = (alone - with_cu).abs().amax(dim=-1)[0]          # per-position max, length LEN_B
for lo, hi in [(0, 10), (10, 50), (50, 100), (100, 200)]:
    print(f"  positions {lo:3d}-{hi:3d}: {d[lo:hi].max().item():.3e}")