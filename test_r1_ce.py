#!/usr/bin/env python
"""Held-out cross-entropy of the TRAINED student, on the same axis as A5b.

A5b measured the UNTRAINED hybrid at 7.9412 and the teacher at 2.3030, on the
first 8 x 2048 held-out tokens.  This measures the R1 checkpoint on the same
stream so the three numbers sit in one column.

Two sample sizes are reported deliberately:
  * first 8 sequences  -> directly comparable to A5b's published 7.9412 / 2.3030
  * all N sequences    -> the headline number, with a standard error

Run:  sbatch test_r1_ce.sbatch          (1 GPU, debug qos, ~10 min at N=64)
"""
import gc
import os
import statistics as st
import sys

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = os.environ["WORK_LUKAS"]
sys.path.insert(0, WORK)
torch.manual_seed(0)

TEACHER = "arcee-ai/AFM-4.5B-Base"
DTYPE, DEV, IMPL = torch.bfloat16, "cuda", "sdpa"     # same as A5b -- do not change
SEQ_LEN = 2048
N_SEQ = int(os.environ.get("N_SEQ", 64))

# label -> path.  Add a row per checkpoint you want on the same axis.
MODELS = [
    ("teacher AFM-4.5B-Base",        TEACHER),
    ("hybrid31, untrained (A5b)",    f"{WORK}/students/afm_kda_hybrid31_init"),
    ("hybrid31, R1 300M tokens",     f"{WORK}/output/r1_stage1_hybrid31"),
]

import transformers
print("transformers", transformers.__version__, "| torch", torch.__version__)
try:
    from importlib.metadata import version
    print("fla", version("flash-linear-attention"))
except Exception:
    pass

# ---------------------------------------------------------------- the tokens
# Identical construction to test_a5b.py, so the first 8 sequences ARE A5b's.
tok = AutoTokenizer.from_pretrained(TEACHER)
ds = load_from_disk(f"{WORK}/datasets/fineweb_edu_3B_heldout")
buf = []
for row in ds:
    buf.extend(tok(row["text"]).input_ids)
    if len(buf) >= SEQ_LEN * N_SEQ:
        break
assert len(buf) >= SEQ_LEN * N_SEQ, f"held-out split too small: {len(buf)} tokens"
ids = torch.tensor(buf[: SEQ_LEN * N_SEQ]).view(N_SEQ, SEQ_LEN).to(DEV)
print(f"evaluating on {N_SEQ} x {SEQ_LEN} = {N_SEQ * SEQ_LEN:,} held-out tokens\n")


@torch.no_grad()
def per_seq_ce(model):
    return [model(ids[i:i + 1], labels=ids[i:i + 1]).loss.float().item()
            for i in range(N_SEQ)]


import math
print(f"{'model':<30} {'CE(first 8)':>12} {'CE(all)':>10} {'+/- se':>8} {'ppl':>9} {'vs teacher':>11}")
print("-" * 86)

ce_teacher_all = None
for label, path in MODELS:
    try:
        m = AutoModelForCausalLM.from_pretrained(
            path, dtype=DTYPE, attn_implementation=IMPL,
            trust_remote_code=True).to(DEV).eval()
    except Exception as e:
        print(f"{label:<30}  LOAD FAILED: {type(e).__name__}: {e}")
        continue

    losses = per_seq_ce(m)
    ce8 = st.mean(losses[:8])
    ce = st.mean(losses)
    se = st.stdev(losses) / math.sqrt(len(losses)) if len(losses) > 1 else float("nan")
    if ce_teacher_all is None:
        ce_teacher_all = ce
    print(f"{label:<30} {ce8:>12.4f} {ce:>10.4f} {se:>8.4f} "
          f"{math.exp(ce):>9.1f} {ce - ce_teacher_all:>+11.4f}")

    del m
    gc.collect()
    torch.cuda.empty_cache()

print("""
Expect: the teacher near 2.3030 and the untrained hybrid near 7.9412 in the
'CE(first 8)' column -- those two are A5b's published numbers and they are
the check that this script is on the same axis.  If they do not reproduce,
the R1 row is not comparable to anything and nothing else here is usable.
""")