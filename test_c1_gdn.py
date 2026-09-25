#!/usr/bin/env python
"""C1 -- the A3 equivalent for the GDN student.

A3 could ask "with every flag at AFM's default, is the student bit-identical
to the teacher?" and expect 0.000e+00.  GDN has no such property: there is no
configuration in which a Gated DeltaNet layer becomes AFM attention.  So the
regression test is built out of four things that are each independently
checkable, in the same spirit:

  1. PARAMETER COUNT against a closed form.  Any geometry mistake moves it.
  2. loading_info -- missing / unexpected key counts against a closed form.
     This re-verifies the whole initialization inventory for free.
  3. FORWARD and BACKWARD, with a gradient on every trainable tensor.
     A5 caught real bugs this way.
  4. HELD-OUT CE at initialization, beside the KDA student's 7.9412.
     Both are "teacher MLPs + random token mixing", so they should land in
     the same region; a wildly different number means the transfer or the
     initialization is wrong, not that GDN is worse.

⚠️ The closed forms in EXPECT below are COMPUTED, not measured.  If the build
disagrees, the arithmetic is what is wrong -- print the measured values and
fix the constants.  That is how 441/108 became trustworthy for KDA.

Run:  sbatch test_c1_gdn.sbatch        (1 GPU; needs ~25 GB for two models)
"""
import math
import os

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = os.environ["WORK_LUKAS"]
GDN = os.environ.get("GDN", f"{WORK}/students/afm_gdn_hybrid31_init")
KDA = os.environ.get("KDA", f"{WORK}/students/afm_kda_hybrid31_init")
TEACHER = os.environ.get(
    "TEACHER",
    f"{WORK}/hf_cache/hub/models--arcee-ai--AFM-4.5B-Base/snapshots/"
    "dab7922f9c868d479b365410304f466b007b1c5a",
)
N_SEQ, SEQ = int(os.environ.get("N_SEQ", 8)), 2048
DEV, DTYPE, IMPL = "cuda", torch.bfloat16, "sdpa"

EXPECT = {
    "params": 5_141_828_536,     # 4,619,184,640 + 27*17,172,648 + 9*6,553,600
    "missing": 360,              # 27 * 13 + 9
    "unexpected": 108,           # 27 * 4
    "gdn_layer": 32_901_288,
    "gdn_delta": 17_172_648,
}
KDA_INIT_CE = 7.9412             # A5b, reproduced by R0 at 7.918


def ok(flag):
    return "✅" if flag else "❌"


print("=" * 78)
print("1. PARAMETER COUNT")
m = AutoModelForCausalLM.from_pretrained(GDN, dtype=DTYPE, trust_remote_code=True)
n = sum(p.numel() for p in m.parameters())
print(f"    class            : {type(m).__name__}")
print(f"    mechanism        : {getattr(m.config, 'mechanism', '<absent>')}")
kinds = [type(l.self_attn).__name__ for l in m.model.layers]
n_gdn = sum(k == "AFMGDNLinearAttention" for k in kinds)
n_attn = sum(k == "AFMKDAAttention" for k in kinds)
print(f"    layers           : {len(kinds)} = {n_gdn} GDN + {n_attn} attention")
print(f"    first attention  : index {kinds.index('AFMKDAAttention') if n_attn else None}")
print(f"    parameters       : {n:,}   expected {EXPECT['params']:,}  {ok(n == EXPECT['params'])}")
if n != EXPECT["params"]:
    print(f"    *** delta {n - EXPECT['params']:+,} -- per converted layer "
          f"{(n - EXPECT['params']) / max(n_gdn, 1):+.1f}")

one = m.model.layers[0].self_attn
if hasattr(one, "attn"):
    k = sum(p.numel() for p in one.attn.parameters())
    print(f"    one GDN layer    : {k:,}   expected {EXPECT['gdn_layer']:,}  "
          f"{ok(k == EXPECT['gdn_layer'])}")

print("\n2. LOADING INFO  (re-read with output_loading_info)")
print("    ⚠️ Only meaningful when loading FROM THE TEACHER, which is what")
print("       build_student.py does. Read build_info.json instead:")
bi = os.path.join(GDN, "build_info.json")
if os.path.exists(bi):
    import json
    info = json.load(open(bi))["loading_info"]
    for key, exp in (("missing_keys", EXPECT["missing"]), ("unexpected_keys", EXPECT["unexpected"])):
        got = info[key] if isinstance(info[key], int) else len(info[key])
        print(f"    {key:16} : {got}   expected {exp}  {ok(got == exp)}")
else:
    print(f"    build_info.json not found at {bi}")

print("\n3. FORWARD AND BACKWARD")
m = m.to(DEV)
m.gradient_checkpointing_disable()
ids = torch.randint(100, 2000, (1, 512), device=DEV)
out = m(input_ids=ids, labels=ids, use_cache=False)
out.loss.backward()
train = [(k, p) for k, p in m.named_parameters() if p.requires_grad]
with_grad = [k for k, p in train if p.grad is not None]
nan = [k for k, p in train if p.grad is not None and not torch.isfinite(p.grad).all()]
print(f"    loss             : {out.loss.item():.4f}")
print(f"    trainable tensors: {len(train)}")
print(f"    with a gradient  : {len(with_grad)}  {ok(len(with_grad) == len(train))}")
print(f"    non-finite grads : {len(nan)}  {ok(not nan)}")
for k in nan[:5]:
    print(f"        !! {k}")
m.zero_grad(set_to_none=True)
del out
torch.cuda.empty_cache()

print("\n4. HELD-OUT CE AT INITIALIZATION")
tok = AutoTokenizer.from_pretrained(TEACHER)
ds = load_from_disk(f"{WORK}/datasets/fineweb_edu_3B_heldout")
buf = []
for row in ds:
    buf.extend(tok(row["text"]).input_ids)
    if len(buf) >= N_SEQ * SEQ:
        break
batch = torch.tensor(buf[: N_SEQ * SEQ], device=DEV).view(N_SEQ, SEQ)


@torch.no_grad()
def ce(model):
    vals = []
    for i in range(batch.shape[0]):
        x = batch[i: i + 1]
        lg = model(input_ids=x, use_cache=False).logits
        vals.append(torch.nn.functional.cross_entropy(
            lg[0, :-1].float(), x[0, 1:]).item())
    return sum(vals) / len(vals)


m.eval()
gdn_ce = ce(m)
print(f"    GDN student, untrained : {gdn_ce:.4f}   ppl {math.exp(gdn_ce):.1f}")
del m
torch.cuda.empty_cache()

for tag, path in [("KDA student, untrained", KDA), ("teacher", TEACHER)]:
    try:
        o = AutoModelForCausalLM.from_pretrained(
            path, dtype=DTYPE, attn_implementation=IMPL,
            trust_remote_code=True).to(DEV).eval()
    except Exception as e:
        print(f"    {tag:23}: LOAD FAILED {type(e).__name__}: {e}")
        continue
    v = ce(o)
    print(f"    {tag:23}: {v:.4f}   ppl {math.exp(v):.1f}")
    del o
    torch.cuda.empty_cache()

print(f"\n    reference: A5b measured the KDA student at {KDA_INIT_CE} on this axis,")
print(f"               and R0 reproduced it at 7.918 inside DistillKit.")

print("""
========================================================================
Reading it

  1-3 must all be ✅.  They are closed forms and a gradient check; there is
  no judgement involved and no tolerance to invent.

  4 is a SANITY BAND, not a pass/fail.  Both students are "the teacher's MLPs
  and embeddings plus randomly-initialised token mixing", so both should land
  near 8 and far below chance (ln 128004 = 11.76).  A GDN number within a few
  tenths of the KDA student's 7.94 means the transfer and the initialisation
  did what they were supposed to.

  ⚠️ A GDN CE meaningfully BETTER than KDA's at initialization would not be
  good news -- it would suggest the two are not comparably initialised, which
  is the confound the drop-in configuration exists to remove.  Investigate
  before celebrating.
========================================================================
""")