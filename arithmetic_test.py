#!/usr/bin/env python
"""Can the student do ONE arithmetic operation?  Isolates computation from
reading comprehension, from chaining, and from termination."""
import os, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
WORK, N = os.environ["WORK_LUKAS"], int(os.environ.get("N", 200))
TEACHER = "arcee-ai/AFM-4.5B-Base"
MODELS = [("teacher", TEACHER),
          ("R1 kda_hybrid", f"{WORK}/output/r1_stage1_hybrid31/checkpoint-4580"),
          ("R2 gdn_hybrid", f"{WORK}/output/r2_stage1_gdn_hybrid31"),
          ("R3 kda_only",   f"{WORK}/output/r3_stage1_kda_only"),
          ("R4 gdn_only",   f"{WORK}/output/r4_stage1_gdn_only")]
SHOTS = ("What is 12 + 7? Answer: 19\nWhat is 45 * 3? Answer: 135\n"
         "What is 200 - 84? Answer: 116\nWhat is 96 / 8? Answer: 12\n")
rng = random.Random(0)
tok = AutoTokenizer.from_pretrained(TEACHER); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token

probs = []
for _ in range(N):
    op = rng.choice("+-*")
    a, b = rng.randint(10, 999), rng.randint(2, 99)
    if op == "-" and b > a: a, b = b, a
    probs.append((f"What is {a} {op} {b}? Answer:", eval(f"{a}{op}{b}")))

for label, path in MODELS:
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True).to("cuda").eval()
    hit = 0
    for i in range(0, N, 16):
        chunk = probs[i:i+16]
        enc = tok([SHOTS + q for q, _ in chunk], return_tensors="pt",
                  padding=True).to("cuda")
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=12, do_sample=False,
                             use_cache=True, pad_token_id=tok.pad_token_id)
        for (q, gold), row in zip(chunk, out):
            txt = tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True)
            got = txt.strip().split()[0].rstrip(".,") if txt.strip() else ""
            hit += (got.lstrip("-").isdigit() and int(got) == gold)
    print(f"{label:16s} single-step accuracy {hit}/{N} = {hit/N:6.1%}")
    del m; torch.cuda.empty_cache()