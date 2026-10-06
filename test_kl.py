#!/usr/bin/env python
"""Held-out CE *and* token-averaged KL for the teacher and every student.

KL replicates distillkit/lossfuncs/kl.py::dense_kl_div exactly, read at
2026-09-29:

    student_logprobs = log_softmax(student_logits.float() / T)
    teacher_logprobs = log_softmax(teacher_logits.float() / T)
    kl = F.kl_div(input=student_logprobs, target=teacher_logprobs,
                  reduction="none", log_target=True).sum(-1)     # per token
    reported = (T**2) * kl.sum() / num_tokens                    # KLDLoss.__call__

  * direction  : KL(teacher || student)  -- forward KL from the teacher
  * temperature: 2.0, applied to BOTH before log_softmax, in fp32
  * T^2 scaling: YES, so the number is 4x the raw mean at T=2
  * reduction  : mean per token (mask is None under packing, so B*T)
  * no shift   : both distributions are read at the same positions

This makes the output directly comparable to the `distillation_loss/2_kl`
figures in the training logs and to Arcee's published ~0.2.

VALIDATION: the teacher appears as its own first row.  Teacher-vs-teacher KL
must print EXACTLY 0.0000.  If it does not, the arithmetic is wrong and no
other row means anything.  (Same check that validated step 5.)

Run:  sbatch test_kl.sbatch        (1 GPU, ~25 min at N_SEQ=64)
"""
import gc
import math
import os
import statistics as st
import sys

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = os.environ["WORK_LUKAS"]
sys.path.insert(0, WORK)
torch.manual_seed(0)

TEACHER = "arcee-ai/AFM-4.5B-Base"
DTYPE, DEV, IMPL = torch.bfloat16, "cuda", "sdpa"    # same axis as A5b / test_r1_ce
SEQ_LEN = 2048
N_SEQ = int(os.environ.get("N_SEQ", 64))
TEMP = float(os.environ.get("TEMP", 2.0))            # must match the YAML's kl temperature
VCHUNK = int(os.environ.get("VCHUNK", 256))          # positions per fp32 chunk
N_LAYERS = 36

# --- replace the MODELS block -------------------------------------------
# R1's TOP-LEVEL directory was overwritten by a misconfigured resume test on
# 2026-09-29 and now holds a 20-step model.  checkpoint-4580 is the real R1
# and is what every benchmark used.  Do not point at the top level again.
MODELS = [
    ("teacher (self -- all must be 0)", TEACHER),
    ("R1 kda_hybrid31",  f"{WORK}/output/r1_stage1_hybrid31/checkpoint-4580"),
    ("R2 gdn_hybrid31",  f"{WORK}/output/r2_stage1_gdn_hybrid31"),
    ("R3 kda_only",      f"{WORK}/output/r3_stage1_kda_only"),
    ("R4 gdn_only",      f"{WORK}/output/r4_stage1_gdn_only"),
]

import transformers
print("transformers", transformers.__version__, "| torch", torch.__version__)
try:
    from importlib.metadata import version
    print("fla", version("flash-linear-attention"))
except Exception:
    pass
print(f"temperature {TEMP}  |  KL(teacher || student), T^2-scaled, mean per token\n")

# ------------------------------------------------------------ the tokens
# Identical construction to test_a5b.py / test_r1_ce.py, so the first 8
# sequences ARE A5b's and the CE column stays on one axis.
tok = AutoTokenizer.from_pretrained(TEACHER)
ds = load_from_disk(f"{WORK}/datasets/fineweb_edu_3B_heldout")
buf = []
for row in ds:
    buf.extend(tok(row["text"]).input_ids)
    if len(buf) >= SEQ_LEN * N_SEQ:
        break
assert len(buf) >= SEQ_LEN * N_SEQ, f"held-out split too small: {len(buf)} tokens"
ids = torch.tensor(buf[: SEQ_LEN * N_SEQ]).view(N_SEQ, SEQ_LEN).to(DEV)
print(f"{N_SEQ} x {SEQ_LEN} = {N_SEQ * SEQ_LEN:,} held-out tokens\n")

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER, dtype=DTYPE, attn_implementation=IMPL,
    trust_remote_code=True).to(DEV).eval()


# --- replace ce_and_kl with this ----------------------------------------
@torch.no_grad()
def ce_kl_hs(student):
    """Per-sequence CE, pooled token-averaged KL, and pooled hs_cosine.

    hs_cosine replicates distillkit/lossfuncs/hidden_state.py::compute_hs_loss
    with kind="cosine", read at 2026-09-29:

        cosine_sim      = F.cosine_similarity(student_h, teacher_h, dim=-1)
        layer_loss      = ((1 - cosine_sim) * mask).sum() / mask.sum()
        total           = mean over the layer_mapping

    `layer_mapping: all` is [(i, i) for i in range(36)], i.e. hidden_states
    indices 0..35 -- the INPUTS to each layer, not the outputs.  Index 0 is the
    embedding output, identical in both models because embed_tokens is frozen,
    so it contributes exactly 0 and drags the mean down by 1/36.  That is also
    true of the training-time number, so the two are comparable.
    No temperature, no scaling: a plain mean of cosine DISTANCES.
    """
    ce, kl_sum, n_tok = [], 0.0, 0
    hs_sum, n_seq = 0.0, 0
    for i in range(N_SEQ):
        x = ids[i:i + 1]
        t_out = teacher(x, output_hidden_states=True)
        s_out = student(x, labels=x, output_hidden_states=True)
        ce.append(s_out.loss.float().item())

        L = x.shape[1]
        for a in range(0, L, VCHUNK):                      # KL, fp32 in slices
            b = min(a + VCHUNK, L)
            sl = torch.log_softmax(s_out.logits[:, a:b].float() / TEMP, dim=-1)
            tl = torch.log_softmax(t_out.logits[:, a:b].float() / TEMP, dim=-1)
            per_tok = F.kl_div(input=sl, target=tl,
                               reduction="none", log_target=True).sum(-1)
            kl_sum += per_tok.sum().item()
            n_tok  += per_tok.numel()
            del sl, tl, per_tok

        layer_total = 0.0                                  # hs_cosine
        for li in range(N_LAYERS):                         # 0..35
            cs = F.cosine_similarity(s_out.hidden_states[li].float(),
                                     t_out.hidden_states[li].float(), dim=-1)
            layer_total += (1.0 - cs).mean().item()        # mask is all-ones here
        hs_sum += layer_total / N_LAYERS
        n_seq  += 1

        del t_out, s_out
    return ce, (TEMP ** 2) * kl_sum / n_tok, hs_sum / n_seq

print(f"{'model':<32} {'CE(8)':>8} {'CE(all)':>9} {'ppl':>8} {'KL':>9} {'hs_cos':>9}")
print("-" * 80)

for label, path in MODELS:
    if path == TEACHER:
        m = teacher
    else:
        try:
            m = AutoModelForCausalLM.from_pretrained(
                path, dtype=DTYPE, attn_implementation=IMPL,
                trust_remote_code=True).to(DEV).eval()
        except Exception as e:
            print(f"{label:<32}  LOAD FAILED: {type(e).__name__}: {e}")
            continue

    ce, kl, hs = ce_kl_hs(m)
    print(f"{label:<32} {st.mean(ce[:8]):>8.4f} {st.mean(ce):>9.4f} "
          f"{math.exp(st.mean(ce)):>8.1f} {kl:>9.4f} {hs:>9.4f}")

    if m is not teacher:
        del m
        gc.collect()
        torch.cuda.empty_cache()

print("""
Reading it
----------
  ROW 1 IS THE VALIDATION.  Teacher against itself must give KL exactly
  0.0000 and CE 2.3030.  Anything else means the arithmetic is wrong.

  The KL column is comparable to `distillation_loss/2_kl` in the training
  logs (R1 ended near 0.314 on the TRAINING stream) and to Arcee's published
  ~0.2 -- but this one is measured on the HELD-OUT split, which is the
  number that belongs in the thesis.

  CE(8) should read 2.3030 for the teacher: that is A5b's published figure
  and the check that this script sits on the same axis as every other CE
  in the project.
""")