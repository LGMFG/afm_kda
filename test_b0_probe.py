#!/usr/bin/env python
"""B0 -- can the INHERITED ArceeModel.forward tolerate fla's Cache?

This is a probe, not a test.  It has no pass/fail: its job is to print what
it sees so that B2's shape is decided from evidence instead of a guess.

Three traps from Track A are deliberately designed around:
  * it never calls dict.get() with a key I guessed -- it DUMPS objects;
  * every branch prints something, including the failure branches;
  * section 3b is a NEGATIVE CONTROL that can falsify my own diagnosis.
"""
import inspect
import sys
import sys
import sys
import os
import traceback

import torch
from transformers import AutoModelForCausalLM

STUDENT = os.environ.get(
    "STUDENT",
    "/leonardo_work/EUHPC_D34_049/WORK_LUKAS/students/afm_kda_hybrid31_init",
)


def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72, flush=True)


def dump(obj, name="object", indent="    "):
    """Print an object's actual contents.  No key names are assumed."""
    print(f"{indent}{name}: {type(obj).__name__}")
    d = getattr(obj, "__dict__", None)
    if not d:
        print(f"{indent}  (no __dict__; dir() = {[a for a in dir(obj) if not a.startswith('_')][:12]})")
        return
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"{indent}  {k}: Tensor{tuple(v.shape)} {v.dtype}")
        elif isinstance(v, (list, tuple)):
            print(f"{indent}  {k}: {type(v).__name__} len={len(v)}"
                  + (f", [0] is {type(v[0]).__name__}" if len(v) else ""))
            if len(v) and isinstance(v[0], dict):
                for kk, vv in v[0].items():
                    shown = (f"Tensor{tuple(vv.shape)} {vv.dtype}"
                             if isinstance(vv, torch.Tensor) else f"{type(vv).__name__} {vv!r}")
                    print(f"{indent}     [0][{kk!r}] = {shown[:70]}")
        elif isinstance(v, dict):
            print(f"{indent}  {k}: dict keys={list(v)[:10]}")
        else:
            print(f"{indent}  {k}: {type(v).__name__} = {str(v)[:60]}")


hr("1. fla's Cache -- what is it, and does it speak transformers' protocol?")

import transformers
from transformers.cache_utils import Cache as HFCache, DynamicCache
from fla.models.utils import Cache as FlaCache

print("transformers :", transformers.__version__)
print("torch        :", torch.__version__)
try:
    from importlib.metadata import version
    print("fla          :", version("flash-linear-attention"))
except Exception as e:
    print("fla          : version lookup failed:", e)

print("\nfla.models.utils.Cache :", FlaCache)
print("MRO                    :", [c.__name__ for c in FlaCache.__mro__])
print("issubclass(fla, HF)    :", issubclass(FlaCache, HFCache))

print("\nmethod inventory (present / defined by fla itself):")
for m in ("update", "get_seq_length", "from_legacy_cache", "to_legacy_cache",
          "__len__", "__getitem__", "get_max_cache_shape", "get_mask_sizes",
          "reorder_cache", "crop", "batch_select_indices", "is_compileable"):
    print(f"  {m:22s} present={str(hasattr(FlaCache, m)):5s} "
          f"own={m in FlaCache.__dict__}")

for m in ("update", "__init__", "from_legacy_cache"):
    try:
        print(f"\nsignature {m}:  {inspect.signature(getattr(FlaCache, m))}")
    except Exception as e:
        print(f"\nsignature {m}: unavailable ({e})")


hr("2. load the student")

model = AutoModelForCausalLM.from_pretrained(
    STUDENT, trust_remote_code=True, dtype=torch.bfloat16,
    attn_implementation="sdpa",
).cuda().eval()

n_kda = sum(1 for l in model.model.layers
            if type(l.self_attn).__name__ == "AFMKDALinearAttention")
print("class                 :", type(model).__name__)
print("params                :", sum(p.numel() for p in model.parameters()))
print("KDA layers            :", n_kda, "of", len(model.model.layers))
print("config.use_cache      :", model.config.use_cache)
print("_supports_cache_class :", getattr(model, "_supports_cache_class", "ABSENT"))
print("_is_stateful          :", getattr(model, "_is_stateful", "ABSENT"))
print("generation eos        :", model.generation_config.eos_token_id)
print("generation bos        :", model.generation_config.bos_token_id)

ids = torch.arange(1000, 1016, device="cuda").unsqueeze(0)   # 1 x 16, fixed
print("input_ids             :", tuple(ids.shape))


hr("3. CONTROL -- what happens today, with transformers' own cache")

with torch.no_grad():
    out = model(input_ids=ids, use_cache=True)
pkv = out.past_key_values
print("returned past_key_values type:", type(pkv).__name__)
for call in ("get_seq_length", "__len__"):
    try:
        print(f"  {call}() ->", getattr(pkv, call)())
    except Exception as e:
        print(f"  {call}() raised {type(e).__name__}: {e}")
dump(pkv, "returned cache")


hr("3b. NEGATIVE CONTROL -- does the bug I claim exists actually exist?")
print("Continuing from a populated cache must DISAGREE with a full recompute.")
print("A large number here confirms the diagnosis.  0.000e+00 would mean")
print("I have misdiagnosed Track B and the plan needs rewriting.\n")

with torch.no_grad():
    o1 = model(input_ids=ids[:, :-1], use_cache=True)
    o2 = model(input_ids=ids[:, -1:], past_key_values=o1.past_key_values, use_cache=True)
    full = model(input_ids=ids, use_cache=False)

d = (o2.logits[0, -1].float() - full.logits[0, -1].float()).abs().max().item()
scale = full.logits[0, -1].float().abs().max().item()
print(f"  cached continuation vs full recompute : {d:.3e}")
print(f"  logit scale                           : {scale:.3e}")
print(f"  argmax agrees                         : "
      f"{o2.logits[0, -1].argmax().item() == full.logits[0, -1].argmax().item()}")


hr("4. THE B0 QUESTION -- hand the inherited forward an fla Cache")

def try_forward(label, factory):
    print(f"\n--- {label}")
    try:
        cache = factory()
    except Exception:
        print("  construction RAISED:")
        traceback.print_exc(limit=4, file=sys.stdout)
        return
    print("  constructed:", type(cache).__name__)
    try:
        with torch.no_grad():
            o = model(input_ids=ids, past_key_values=cache, use_cache=True)
    except Exception:
        print("  forward RAISED:")
        traceback.print_exc(limit=10, file=sys.stdout)
        return
    print("  *** SURVIVED the forward ***")
    print("  returned pkv type:", type(o.past_key_values).__name__)
    print("  logits finite    :", bool(torch.isfinite(o.logits).all()))
    dump(o.past_key_values, "cache after forward")
    print("  (the KDA layers should have stored NOTHING yet -- the adapter")
    print("   still hardcodes past_key_values=None.  That is B4's job.)")

try_forward("empty FlaCache()", lambda: FlaCache())
try_forward("FlaCache.from_legacy_cache(None)", lambda: FlaCache.from_legacy_cache(None))
try_forward("FlaCache.from_legacy_cache(DynamicCache())",
            lambda: FlaCache.from_legacy_cache(DynamicCache()))


hr("VERDICT -- read this, then choose B2's shape")
print("""
If section 4 SURVIVED:
    B2 is the easy version.  Convert in AFMKDAModel.forward, conditional on
    use_cache, and call super().forward() unchanged.  Proceed to B1.

If section 4 RAISED inside ArceeModel.forward or a masking utility:
    The inherited forward does transformers-native cache bookkeeping that
    fla's Cache cannot answer.  Read the traceback for WHICH method it
    wanted.  Two options, in order of preference:
      (a) subclass fla's Cache and add just that method;
      (b) keep transformers' DynamicCache for the 9 attention layers and give
          the 27 KDA layers a SEPARATE fla Cache carried in kwargs -- two
          caches, one model.  Uglier than the reference, but it leaves
          ArceeModel.forward untouched, which is what A3 depends on.

If section 3b printed 0.000e+00:
    STOP.  The KDA layers are somehow not being reached by the cached path,
    or the model is recomputing internally.  Do not write B2 until that is
    understood -- the whole of Track B rests on this number being large.
""")
