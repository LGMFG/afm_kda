#!/usr/bin/env python
"""B7 -- does the cache still work in a BATCH with LEFT PADDING?

This is how lm-eval generates: several prompts of different lengths stacked
into one tensor, shorter ones padded on the left, with a 0/1 attention_mask
marking the padding.

Reference for each row : that row ALONE, unpadded, one full pass, no cache.
Test                   : all four rows together, left-padded, cached stepwise.

⚠️ A failure here will be LOUD, not subtle.  If the padding mask never
reaches the KDA layers, their recurrent state ingests the pad tokens as if
they were text, and the padded rows come out wrong by O(1-10), not by a ulp.
The unpadded row (the longest one) will look fine, which is exactly the trap.

The TEACHER is run through the identical protocol as a control: batching and
padding perturb the arithmetic even in a correct implementation, so the
question is not "is the student's difference zero" but "is it the same size
as the teacher's".

Run:  sbatch test_b7_batched.sbatch      (1 GPU, debug qos)
"""
import math
import os
import sys

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = os.environ["WORK_LUKAS"]
STUDENT = os.environ.get("MODEL", f"{WORK}/output/r1_stage1_hybrid31")
TEACHER = "arcee-ai/AFM-4.5B-Base"
LENS = [int(x) for x in os.environ.get("LENS", "120,180,240,300").split(",")]
N_NEW = int(os.environ.get("N_NEW", 32))
DTYPE, DEV = torch.bfloat16, "cuda"
IMPL = os.environ.get("IMPL", "sdpa")   # try "eager" if sdpa rejects the mask

tok = AutoTokenizer.from_pretrained(TEACHER)
PAD = tok.eos_token_id
ds = load_from_disk(f"{WORK}/datasets/fineweb_edu_3B_heldout")

# One long stream, cut into rows of DIFFERENT lengths so the padding is real.
need = sum(LENS) + len(LENS) * N_NEW
buf = []
for row in ds:
    buf.extend(tok(row["text"]).input_ids)
    if len(buf) >= need:
        break

rows, pos = [], 0
for L in LENS:
    rows.append(buf[pos: pos + L + N_NEW])
    pos += L + N_NEW

B, W = len(LENS), max(LENS)
padded = torch.full((B, W), PAD, dtype=torch.long, device=DEV)
mask = torch.zeros((B, W), dtype=torch.long, device=DEV)
for i, L in enumerate(LENS):
    padded[i, W - L:] = torch.tensor(rows[i][:L], device=DEV)   # LEFT padding
    mask[i, W - L:] = 1
cont = torch.stack([torch.tensor(r[L: L + N_NEW], device=DEV)
                    for r, L in zip(rows, LENS)])

print(f"batch {B} | lengths {LENS} | padded width {W} | {N_NEW} forced tokens | impl={IMPL}")
print(f"left padding per row: {[W - L for L in LENS]}\n")


def ulp(x):
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else float("nan")


@torch.no_grad()
def reference(m, i):
    """Row i alone, no padding, no cache -- the ground truth for that row."""
    L = LENS[i]
    ids = torch.tensor(rows[i][: L + N_NEW], device=DEV).unsqueeze(0)
    return m(input_ids=ids, use_cache=False).logits[0, L - 1: L - 1 + N_NEW].float()


@torch.no_grad()
def batched(m):
    """All rows together, left-padded, cached, fed the same continuations."""
    out = m(input_ids=padded, attention_mask=mask, use_cache=True)
    pkv = out.past_key_values
    step_mask = mask
    logits = [out.logits[:, -1]]
    for k in range(N_NEW - 1):
        step_mask = torch.cat(
            [step_mask, torch.ones((B, 1), dtype=torch.long, device=DEV)], dim=1)
        out = m(input_ids=cont[:, k: k + 1], attention_mask=step_mask,
                past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        logits.append(out.logits[:, -1])
    return torch.stack(logits, dim=1).float()      # [B, N_NEW, V]


for tag, path in [("TEACHER (control)", TEACHER), ("STUDENT hybrid31", STUDENT)]:
    print(f"=== {tag}")
    try:
        m = AutoModelForCausalLM.from_pretrained(
            path, dtype=DTYPE, attn_implementation=IMPL,
            trust_remote_code=True).to(DEV).eval()
    except Exception as e:
        print(f"    LOAD FAILED: {type(e).__name__}: {e}\n")
        continue

    try:
        bat = batched(m)
    except Exception as e:
        # A crash HERE is itself the finding -- and it must not stop the other
        # model from running, because "does the teacher crash too?" is the
        # whole diagnostic.
        import traceback
        print(f"    *** batched forward RAISED {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stdout, limit=6)
        print()
        del m
        torch.cuda.empty_cache()
        continue

    print(f"    {'row':>4} {'len':>5} {'pad':>5} {'worst |Δ|':>12} {'ulp':>7} "
          f"{'argmax':>9}")
    for i in range(B):
        ref = reference(m, i)
        d = (ref - bat[i]).abs()
        agree = (ref.argmax(-1) == bat[i].argmax(-1)).sum().item()
        sc = ref.abs().max().item()
        print(f"    {i:>4} {LENS[i]:>5} {W - LENS[i]:>5} {d.max().item():>12.3e} "
              f"{d.max().item() / ulp(sc):>6.1f}x {agree:>5}/{N_NEW}")
    print()
    del m
    torch.cuda.empty_cache()

print("""
Reading it
----------
Compare the STUDENT rows to the TEACHER rows, not to zero.  Batching and
left padding change the arithmetic in any implementation.

  student rows ~ teacher rows, argmax agreement high
      -> B7 passes.  Set use_cache: true on the eval checkpoints and go to B8.

  the PADDED rows are orders of magnitude worse than the unpadded one
      -> the padding mask is not reaching the KDA layers.  Predicted cause:
         AFMKDAModel.forward hands the 2-D mask to super().forward(), which
         turns it into a 4-D causal mask, and AFMKDALinearAttention drops any
         mask that is not 2-D.  Fix: stash the raw mask in kwargs before
         calling super(), the same way cu_seqlens is stashed, and prefer it
         in the adapter.
""")