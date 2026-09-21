#!/usr/bin/env python
"""B6d -- is the cached path LOGICALLY equal, or only approximately equal?

B6's teacher-forced comparison in bf16 disagrees by 2-3 ulp.  Two explanations
fit that, and they have opposite consequences:

  (1) rounding.  The two paths run DIFFERENT KERNELS -- fla switches to
      fused_recurrent_kda whenever q_len <= 64 and chunk_kda otherwise
      (kda.py: `mode = "fused_recurrent" if (q_len <= 64 and not self.training)`),
      so a 428-token pass and a 300-token pass plus 127 single-token steps are
      not the same arithmetic at all.  A few ulp is then expected and harmless.
  (2) a logic bug in the bridge, hidden under rounding.

⭐ Running the SAME comparison in float32 separates them without inventing a
tolerance.  Rounding shrinks by ~2^16; a logic bug does not shrink at all.

  expected under (1): ~1e-5 or smaller
  expected under (2): O(0.1 - 10), the same as in bf16

Also reports ARGMAX AGREEMENT over the forced positions, which is the property
that actually matters for generation, and a TEACHER CONTROL -- the same
protocol on the unmodified all-attention teacher, whose cache is transformers'
own and is not in question.

Run:  sbatch test_b6d_precision.sbatch     (1 GPU; fp32 needs ~20 GB for weights)
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
N_PROMPT = int(os.environ.get("N_PROMPT", 128))
N_NEW = int(os.environ.get("N_NEW", 32))
DEV, IMPL = "cuda", "sdpa"


def ulp(x):
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else float("nan")


tok = AutoTokenizer.from_pretrained(TEACHER)
ds = load_from_disk(f"{WORK}/datasets/fineweb_edu_3B_heldout")
buf = []
for row in ds:
    buf.extend(tok(row["text"]).input_ids)
    if len(buf) >= N_PROMPT + N_NEW:
        break
full = torch.tensor(buf[: N_PROMPT + N_NEW], device=DEV).unsqueeze(0)
print(f"{N_PROMPT} prompt + {N_NEW} forced tokens of real text\n")


@torch.no_grad()
def forced_gap(m):
    """Max |Δlogit| between one full pass and prompt-plus-stepwise, same tokens."""
    prompt, cont = full[:, :N_PROMPT], full[:, N_PROMPT:]
    a = m(input_ids=full, use_cache=False).logits[0, N_PROMPT - 1: N_PROMPT - 1 + N_NEW]

    out = m(input_ids=prompt, use_cache=True)
    pkv = out.past_key_values
    b = [out.logits[0, -1]]
    for k in range(N_NEW - 1):
        out = m(input_ids=cont[:, k: k + 1], past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        b.append(out.logits[0, -1])
    b = torch.stack(b)

    a32, b32 = a.float(), b.float()
    d = (a32 - b32).abs()
    agree = (a32.argmax(-1) == b32.argmax(-1)).sum().item()
    return dict(worst=d.max().item(),
                median_pos=d.max(dim=-1).values.median().item(),
                first=d[0].max().item(), last=d[-1].max().item(),
                scale=a32.abs().max().item(),
                agree=agree, n=a32.shape[0])


def report(tag, r):
    u = ulp(r["scale"])
    print(f"  {tag}")
    print(f"    worst |Δlogit|      : {r['worst']:.3e}   ({r['worst'] / u:.1f} ulp at scale {r['scale']:.1f})")
    print(f"    median per position : {r['median_pos']:.3e}")
    print(f"    first -> last pos   : {r['first']:.3e} -> {r['last']:.3e}")
    print(f"    argmax agreement    : {r['agree']}/{r['n']}")
    return r["worst"]


results = {}
for tag, path, dtype in [
    ("STUDENT bf16  (kernels differ: chunk vs fused_recurrent)", STUDENT, torch.bfloat16),
    ("STUDENT fp32  <- the decisive one", STUDENT, torch.float32),
    ("TEACHER bf16  (control: all attention, transformers' own cache)", TEACHER, torch.bfloat16),
]:
    print(f"=== {tag}")
    try:
        m = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, attn_implementation=IMPL,
            trust_remote_code=True).to(DEV).eval()
    except torch.cuda.OutOfMemoryError:
        print("    OOM -- lower N_PROMPT, or skip the fp32 row and rely on the teacher control\n")
        torch.cuda.empty_cache()
        continue
    except Exception as e:
        print(f"    LOAD FAILED: {type(e).__name__}: {e}\n")
        continue
    results[tag.split()[1]] = report(tag, forced_gap(m))
    print()
    del m
    torch.cuda.empty_cache()

print("=" * 72)
bf = results.get("bf16")
print("""VERDICT

Look at the fp32 row.

  worst |Δlogit| ~1e-5 or below
      -> the bridge is LOGICALLY CORRECT.  Everything seen in bf16 is the
         difference between chunk_kda and fused_recurrent_kda, which fla
         switches between by sequence length on purpose.  Track B passes;
         record the bf16 figure as a kernel property, not an error.

  worst |Δlogit| still ~0.1 or larger
      -> a real bug that rounding was hiding.  Suspect the conv_state or the
         offset passed to update_layer_cache, and do NOT proceed to B7.

The teacher row calibrates: if the unmodified teacher also disagrees with
itself by a couple of ulp in bf16, that magnitude is normal for this model
and this dtype and says nothing about KDA.
""")