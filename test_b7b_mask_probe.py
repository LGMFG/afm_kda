#!/usr/bin/env python
"""B7b -- WHY does the student crash in SDPA where the teacher does not?

B7 established the important half already: the TEACHER passes the identical
batched, left-padded, cached protocol on all four rows (1.2-1.8 ulp, 31-32/32
argmax).  So the protocol is sound and the failure is specific to our model.

Since it is an SDPA *bias* error, it is in the nine attention layers, and the
suspicion is a second method with the same bug we already fixed once:

    get_seq_length()   -- we OVERRODE it, because DynamicCache reads layer 0
                          and layer 0 of a hybrid is a KDA layer that stores
                          nothing (B0 measured it returning 0).
    get_mask_sizes()   -- we did NOT override it.  create_causal_mask() calls
                          it to decide how wide the mask should be.  If it is
                          also answering for layer 0, the mask is built for a
                          length the attention layers do not have.

This probe does not assume that.  It PRINTS what each method answers, per
layer, for the student and the teacher, and it dumps the shape / stride /
contiguity of the mask that actually reaches an attention layer.

Run:  sbatch test_b7b_mask_probe.sbatch     (1 GPU, debug qos, ~5 min)
"""
import os
import sys
import traceback

import torch
from transformers import AutoModelForCausalLM

WORK = os.environ["WORK_LUKAS"]
STUDENT = os.environ.get("MODEL", f"{WORK}/output/r1_stage1_hybrid31")
TEACHER = "arcee-ai/AFM-4.5B-Base"
DTYPE, DEV, IMPL = torch.bfloat16, "cuda", os.environ.get("IMPL", "sdpa")

B, W, PADS = 4, 300, [180, 120, 60, 0]
ids = torch.randint(100, 2000, (B, W), device=DEV)
mask = torch.ones((B, W), dtype=torch.long, device=DEV)
for i, p in enumerate(PADS):
    mask[i, :p] = 0


def hr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72, flush=True)


def install_probe(m, tag):
    """Print the mask each attention layer actually receives, once."""
    seen = {"n": 0}

    def pre(mod, args, kwargs):
        if seen["n"] >= 2:
            return
        seen["n"] += 1
        am = kwargs.get("attention_mask")
        if am is None and len(args) > 2:
            am = args[2]
        hs = kwargs.get("hidden_states", args[0] if args else None)
        print(f"    [{tag}] q_len={tuple(hs.shape) if hs is not None else '?'} "
              f"mask={'None' if am is None else tuple(am.shape)}"
              + ("" if am is None else
                 f" stride={am.stride()} contiguous={am.is_contiguous()} dtype={am.dtype}"))

    for i, layer in enumerate(m.model.layers):
        if type(layer.self_attn).__name__ in ("AFMKDAAttention", "ArceeAttention"):
            layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True)
            break
    return seen


for tag, path in [("TEACHER", TEACHER), ("STUDENT", STUDENT)]:
    hr(f"{tag}   ({path.split('/')[-1]})")
    try:
        m = AutoModelForCausalLM.from_pretrained(
            path, dtype=DTYPE, attn_implementation=IMPL,
            trust_remote_code=True).to(DEV).eval()
    except Exception:
        traceback.print_exc(file=sys.stdout)
        continue

    kinds = [type(l.self_attn).__name__ for l in m.model.layers]
    print("layer 0 is:", kinds[0], "| first attention layer at index:",
          next((i for i, k in enumerate(kinds) if k != "AFMKDALinearAttention"), None))

    install_probe(m, tag)

    print("\n--- prompt forward (q_len=300)")
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=mask, use_cache=True)
    pkv = out.past_key_values
    print("    cache:", type(pkv).__name__, "| get_seq_length() ->",
          pkv.get_seq_length(), "(expect 300)")

    # What does each per-layer accessor answer?  No key names guessed.
    cache_pos = torch.tensor([W], device=DEV)
    print("\n--- per-layer answers  (kv_length, kv_offset) from get_mask_sizes")
    for i in (0, 1, 2, 3, 4):
        try:
            print(f"    layer {i} ({kinds[i]:>22}): get_mask_sizes -> "
                  f"{pkv.get_mask_sizes(cache_pos, i)}   "
                  f"get_seq_length(i) -> "
                  f"{super(type(pkv), pkv).get_seq_length(i) if hasattr(pkv, 'attn_layer_idx') else pkv.get_seq_length(i)}")
        except Exception as e:
            print(f"    layer {i} ({kinds[i]:>22}): raised {type(e).__name__}: {e}")

    print("\n--- one decode step (q_len=1)  <- this is where B7 died")
    step_mask = torch.cat([mask, torch.ones((B, 1), dtype=torch.long, device=DEV)], 1)
    nxt = torch.randint(100, 2000, (B, 1), device=DEV)
    try:
        with torch.no_grad():
            m(input_ids=nxt, attention_mask=step_mask, past_key_values=pkv, use_cache=True)
        print("    SURVIVED")
    except Exception:
        print("    RAISED -- full traceback, no limit:")
        traceback.print_exc(file=sys.stdout)      # limit=None: the deep frames matter

    del m
    torch.cuda.empty_cache()

hr("What to look for")
print("""
Compare the two 'per-layer answers' blocks.

  STUDENT layer 0 reports a kv_length of ~0/1 while the teacher's layer 0
  reports ~300
      -> create_causal_mask is sizing the mask from a KDA layer that stores
         nothing.  Fix: override get_mask_sizes on AFMKDACache the same way
         get_seq_length is already overridden -- answer for attn_layer_idx.

  the mask reaching the attention layer is non-contiguous with a stride whose
  last entry is not 1
      -> that is the tensor SDPA rejects; the shape tells you which dimension
         was sliced.

  both models answer identically and the student still raises
      -> the difference is in our COPY of modeling_arcee.py versus the
         installed one.  diff them.
""")