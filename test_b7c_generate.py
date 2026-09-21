#!/usr/bin/env python
"""B7c -- one real generate() call.

Every test on this track hand-rolled the decode loop.  lm-eval calls
model.generate(), which reads generation_config, builds its own cache, and
manages cache_position itself.  This is the first time that code path runs end
to end, and it is the last gap between "our loop works" and "the harness will
work".  Five minutes on the debug queue.

It also double-checks the use_cache flip by printing what the loaded model
actually believes, rather than what the file on disk says.

Run:  sbatch test_b7c_generate.sbatch      (1 GPU, debug qos)
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = os.environ["WORK_LUKAS"]
STUDENT = os.environ.get("MODEL", f"{WORK}/output/r1_stage1_hybrid31")
TEACHER = "arcee-ai/AFM-4.5B-Base"
IMPL = os.environ.get("IMPL", "sdpa")
N_NEW = int(os.environ.get("N_NEW", 48))

# Different lengths on purpose, so the left padding in the batch is real.
PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of about",
    "In July 1969, three astronauts travelled to the Moon aboard a spacecraft named",
]

tok = AutoTokenizer.from_pretrained(TEACHER)
tok.padding_side = "left"                 # how lm-eval batches generation
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

for tag, path in [("TEACHER (control)", TEACHER), ("STUDENT hybrid31", STUDENT)]:
    print("=" * 72)
    print(tag, flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation=IMPL,
        trust_remote_code=True).to("cuda").eval()

    print(f"    config.use_cache            : {m.config.use_cache}")
    print(f"    generation_config.use_cache : {m.generation_config.use_cache}")
    kinds = [type(l.self_attn).__name__ for l in m.model.layers]
    print(f"    layers                      : {len(kinds)}, "
          f"{sum(k == 'AFMKDALinearAttention' for k in kinds)} KDA / "
          f"{sum(k != 'AFMKDALinearAttention' for k in kinds)} attention")

    enc = tok(PROMPTS, return_tensors="pt", padding=True).to("cuda")
    print(f"    batch shape                 : {tuple(enc.input_ids.shape)}, "
          f"pads per row {(enc.attention_mask == 0).sum(1).tolist()}")

    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=N_NEW, do_sample=False,
                         pad_token_id=tok.pad_token_id)

    for p, o in zip(PROMPTS, out):
        print(f"\n    > {p}")
        print(f"      {tok.decode(o, skip_special_tokens=True)!r}")
    print()
    del m
    torch.cuda.empty_cache()

print("""
========================================================================
Reading it

You are reading for COHERENCE, not for agreement with the teacher.  Greedy
decoding is a chaotic map and B6c already showed the two diverge on near-ties;
identical text is not the bar and never was.

  PASS  the student writes fluent, on-topic English for all three prompts,
        in one batch, with left padding, and config.use_cache prints True.

  FAIL  repetition loops, or text that degenerates after a few tokens, or --
        the one to watch -- the two SHORT prompts degenerate while the long
        one is fine.  That is pad contamination of the recurrent state, which
        B7's ulp comparison would have been too gentle to catch at 32 tokens.
========================================================================
""")