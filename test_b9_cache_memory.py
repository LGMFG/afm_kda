#!/usr/bin/env python
"""B9a -- inference state size versus context length, MEASURED.

Since B4' the KDA layers actually populate their recurrent state, so BOTH
halves of the inference state are now measured rather than computed:

  attention KV  : grows linearly in sequence length
  KDA state     : constant in sequence length -- the architectural claim

⚠️ The cache object carries the fla state as an attribute (AFMKDACache.fla),
so a naive walk of past_key_values returns KV + KDA together.  This script
separates them and checks each against its own closed form.  An earlier
version did not, which is why its closed-form check flagged MISMATCH the
first time the bridge worked -- the check doing exactly its job.

⚠️ GEOMETRY, corrected against measurement 2026-09-20:
  recurrent state : num_heads x head_dim x head_dim, in FLOAT32  (not bf16)
  conv state      : 3 x hidden_size x conv_size,     in bf16
  -> 20*128*128*4 + 3*2560*4*2 = 1,372,160 B = 1.3086 MB per KDA layer
Predicted 18.06 MB for 27 layers under a bf16 assumption; measured 35.33.
The fp32 recurrent state is the difference.

Run:  sbatch test_b9_cache_memory.sbatch      (1 GPU, debug qos)
"""
import gc
import os

import torch
from transformers import AutoModelForCausalLM

WORK = os.environ["WORK_LUKAS"]
DTYPE, DEV, IMPL = torch.bfloat16, "cuda", "sdpa"
LENGTHS = [int(x) for x in os.environ.get(
    "LENGTHS", "512,1024,2048,4096,8192,16384,32768").split(",")]

MODELS = [
    ("teacher (36 attn)",   "arcee-ai/AFM-4.5B-Base"),
    ("hybrid 3:1 (9 attn)", f"{WORK}/output/r1_stage1_hybrid31"),
    ("kda_only (0 attn)",   f"{WORK}/students/afm_kda_only_init"),
]

MB = 1024 ** 2


def cache_bytes(obj, seen=None):
    """Sum every tensor reachable from a cache object.  Nothing is assumed."""
    if obj is None:
        return 0
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple, set)):
        return sum(cache_bytes(x, seen) for x in obj)
    if isinstance(obj, dict):
        return sum(cache_bytes(x, seen) for x in obj.values())
    d = getattr(obj, "__dict__", None)
    if d:
        return sum(cache_bytes(x, seen) for x in d.values())
    return 0


def kda_state_bytes(cfg, n_kda):
    h = getattr(cfg, "kda_num_heads", 20)
    d = getattr(cfg, "kda_head_dim", 128)
    conv = getattr(cfg, "kda_conv_size", 4)
    return n_kda * (h * d * d * 4 + 3 * cfg.hidden_size * conv * 2)


def dump_state_dtypes(fla_cache, limit=8):
    """Print what is actually in there -- dtype inferred from bytes is a guess."""
    print("      fla state contents (first layer):")
    shown = 0
    stack = [fla_cache]
    seen = set()
    while stack and shown < limit:
        o = stack.pop(0)
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, torch.Tensor):
            print(f"        Tensor{tuple(o.shape)} {o.dtype} "
                  f"= {o.numel() * o.element_size() / 1024:.1f} KB")
            shown += 1
        elif isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, torch.Tensor):
                    print(f"        [{k!r}] Tensor{tuple(v.shape)} {v.dtype} "
                          f"= {v.numel() * v.element_size() / 1024:.1f} KB")
                    shown += 1
                else:
                    stack.append(v)
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
        elif getattr(o, "__dict__", None):
            stack.extend(o.__dict__.values())


print(f"{'model':<22} {'T':>7} {'attn KV':>10} {'KDA state':>11} "
      f"{'TOTAL (MB)':>11} {'vs teacher':>11} {'peak fwd':>10}")
print("-" * 92)

teacher_total = {}
for label, path in MODELS:
    try:
        m = AutoModelForCausalLM.from_pretrained(
            path, dtype=DTYPE, attn_implementation=IMPL,
            trust_remote_code=True).to(DEV).eval()
    except Exception as e:
        print(f"{label:<22}  LOAD FAILED: {type(e).__name__}: {e}")
        continue

    cfg = m.config
    n_kda = sum(1 for l in m.model.layers
                if type(l.self_attn).__name__ == "AFMKDALinearAttention")
    n_attn = len(m.model.layers) - n_kda
    dumped = False

    for T in LENGTHS:
        ids = torch.zeros(1, T, dtype=torch.long, device=DEV)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        try:
            with torch.no_grad():
                # use_cache must be passed EXPLICITLY: config.use_cache is False.
                out = m(input_ids=ids, use_cache=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{label:<22} {T:>7}   OOM on the forward "
                  f"(cache would be {(n_attn * 2 * T * cfg.num_key_value_heads * cfg.head_dim * 2 + kda_state_bytes(cfg, n_kda)) / MB:.1f} MB)")
            torch.cuda.empty_cache()
            continue
        peak = (torch.cuda.max_memory_allocated() - base) / MB

        pkv = out.past_key_values
        fla = getattr(pkv, "fla", None)
        fla_b = cache_bytes(fla)                       # KDA recurrent + conv
        kv_b = cache_bytes(pkv) - fla_b                # attention only
        total = (kv_b + fla_b) / MB
        if label.startswith("teacher"):
            teacher_total[T] = total
        ratio = f"{teacher_total.get(T, float('nan')) / total:>10.2f}x" if total else ""
        print(f"{label:<22} {T:>7} {kv_b / MB:>10.1f} {fla_b / MB:>11.2f} "
              f"{total:>11.1f} {ratio:>11} {peak:>10.0f}")

        if fla_b and not dumped:
            dump_state_dtypes(fla)
            dumped = True
        del out
        gc.collect()

    # Two closed forms, checked separately -- a single combined check cannot
    # say WHICH half moved.
    T = LENGTHS[-1]
    kv_pred = n_attn * 2 * T * cfg.num_key_value_heads * cfg.head_dim * 2 / MB
    fla_pred = kda_state_bytes(cfg, n_kda) / MB
    print(f"{'':<22} check KV : {n_attn} x 2 x {T} x {cfg.num_key_value_heads} "
          f"x {cfg.head_dim} x 2B = {kv_pred:.1f} MB")
    print(f"{'':<22} check KDA: {n_kda} x (20*128*128*4 + 3*{cfg.hidden_size}*4*2) "
          f"= {fla_pred:.2f} MB\n")

    del m
    gc.collect()
    torch.cuda.empty_cache()

print("""
Reading it
----------
'attn KV' grows linearly in T.  'KDA state' does not move at all.

The model-level ratio is bounded by how many attention layers survive: a 3:1
hybrid keeps 9 of 36.  The KDA state is NOT free -- it is a fixed 1.31 MB per
layer -- so below a crossover around T = 670 tokens a converted model carries
MORE state than the teacher, not less.  Report the crossover; it is honest and
it is where the argument actually lives.
""")