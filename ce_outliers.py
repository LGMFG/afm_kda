#!/usr/bin/env python
"""Why do two measurements of R5 disagree?

  DistillKit's own held-out eval:  R5 eval_loss 0.5715 vs R1 0.5686  -> level
  test_r1_ce_math.py:              R5 web CE 2.7632 vs R1 2.4559     -> +0.31 nats
                                   paired se 0.152 vs R1's 0.007     -> 20x the spread

A 20x larger paired spread means a FEW sequences are catastrophically bad,
not that R5 is uniformly worse.  R5 also diverged repeatedly during training
(eval_loss 0.64 -> 1.74 -> 1.06 -> 1.76 before recovering; grad_norm peaked
at 83,070).  So the question is whether the recovered model carries damage
that shows up only under some conditions.  Five measurements, one job:

  1. per-SEQUENCE CE, R5 vs R1 vs teacher   -> how many sequences, how bad
  2. per-POSITION CE (8 bins over 2048)      -> does R5's excess GROW with
                                                 position?  (a recurrent state
                                                 that accumulates instead of
                                                 forgetting does exactly that)
  3. sdpa vs flash_attention_2               -> DistillKit trains and evals
                                                 with FA2; test_r1_ce_math uses
                                                 sdpa.  If R5 is fine under FA2
                                                 the disagreement is the kernel
  4. per-LAYER hidden-state magnitude on the worst sequence -> which layer
  5. WEIGHT drift R5 vs R1, from the safetensors on CPU    -> which parameters

The sequences are built exactly as test_r1_ce_math.py builds them (same
documents, same order, same concatenation), so the R1 and teacher rows must
reproduce 2.4559 and 2.2963 on web.  If they do not, stop.

Run:  sbatch ce_outliers.sbatch     (1 GPU, debug, ~15-20 min)
"""
import glob
import math
import os

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = os.environ["WORK_LUKAS"]
TEACHER = (f"{WORK}/hf_cache/hub/models--arcee-ai--AFM-4.5B-Base/snapshots/"
           "dab7922f9c868d479b365410304f466b007b1c5a")
R1 = os.environ.get("R1", f"{WORK}/output/r1_stage1_hybrid31/checkpoint-4580")
R5 = os.environ.get("R5", f"{WORK}/output/r5_stage1_kda_hybrid31_1ep")
CORPORA = {"web": f"{WORK}/datasets/fineweb_edu_3B_heldout"}
if os.environ.get("TULU3"):
    CORPORA["tulu3"] = os.environ["TULU3"]
IMPLS = os.environ.get("IMPLS", "sdpa,flash_attention_2").split(",")
assert "sdpa" in IMPLS, "the report is anchored on sdpa (test_r1_ce_math.py's setting)"
SEQ, N_SEQ, NBIN = 2048, int(os.environ.get("N_SEQ", 64)), 8
DEV, DTYPE = "cuda", torch.bfloat16

tok = AutoTokenizer.from_pretrained(TEACHER)


# ---- batch construction: identical to test_r1_ce_math.py -----------------
COLUMN_SETS = [("messages",), ("query", "response"), ("question", "answer"),
               ("problem", "solution"), ("text",)]


def pick_columns(names):
    for cand in COLUMN_SETS:
        if all(c in names for c in cand):
            return list(cand)
    raise SystemExit(f"cannot tell which columns hold text: {names}")


def row_text(row, cols):
    if cols == ["messages"]:
        parts = [str(m.get("content", "")) if isinstance(m, dict) else str(m)
                 for m in row["messages"]]
        return "\n".join(p for p in parts if p)
    return "\n".join(str(row[c]) for c in cols if row[c])


def build_batch(path):
    ds = load_from_disk(path)
    if hasattr(ds, "keys"):
        ds = ds["train" if "train" in ds else list(ds.keys())[0]]
    cols = pick_columns(ds.column_names)
    need, buf = N_SEQ * SEQ + 1, []
    for row in ds:
        buf.extend(tok(row_text(row, cols)).input_ids)
        if len(buf) >= need:
            break
    return torch.tensor(buf[: N_SEQ * SEQ]).view(N_SEQ, SEQ)


# ---- measurement ----------------------------------------------------------
@torch.no_grad()
def token_ce(model, batch):
    """[N_SEQ, SEQ-1] per-token CE in fp32, plus non-finite logit counts."""
    out, bad = [], 0
    for i in range(batch.shape[0]):
        ids = batch[i:i + 1].to(DEV)
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
        bad += (~torch.isfinite(logits)).any(-1).sum().item()
        out.append(torch.nn.functional.cross_entropy(
            logits, ids[0, 1:], reduction="none").cpu())
    return torch.stack(out), bad


@torch.no_grad()
def layer_magnitudes(model, ids):
    hs = model(input_ids=ids.to(DEV), use_cache=False,
               output_hidden_states=True).hidden_states
    return [(h.float().abs().max().item(), h.float().pow(2).mean().sqrt().item()) for h in hs]


def load(path, impl):
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=DTYPE, attn_implementation=impl,
        trust_remote_code=True).to(DEV).eval()


print(f"{N_SEQ} x {SEQ} tokens per corpus;  R1 = {R1}\n                              R5 = {R5}\n")
batches = {c: build_batch(p) for c, p in CORPORA.items()}
ce, nonfinite = {}, {}
runs = [("teacher", TEACHER, "sdpa")] + [(m, p, i) for i in IMPLS for m, p in (("r1", R1), ("r5", R5))]

worst_seq, r5_mags = None, None
for name, path, impl in runs:
    key = f"{name}/{impl}"
    print(f"=== {key}", flush=True)
    m = load(path, impl)
    for c, b in batches.items():
        ce[(key, c)], nonfinite[(key, c)] = token_ce(m, b)
        v = ce[(key, c)].mean(1)
        print(f"    {c:6} CE {v.mean():.4f}   median-seq {v.median():.4f}   "
              f"worst-seq {v.max():.3f}   non-finite logit rows {nonfinite[(key, c)]}", flush=True)
    if name == "r5" and impl == "sdpa":            # r1/sdpa has already run
        d = ce[("r5/sdpa", "web")].mean(1) - ce[("r1/sdpa", "web")].mean(1)
        worst_seq = int(d.argmax())
        try:
            r5_mags = layer_magnitudes(m, batches["web"][worst_seq:worst_seq + 1])
        except Exception as e:                      # output_hidden_states unsupported
            print(f"    (layer magnitudes unavailable: {e})")
            r5_mags = None
    del m
    torch.cuda.empty_cache()
    print()

# R1 layer magnitudes on the same worst sequence (reload once, sdpa)
r1_mags = None
if r5_mags is not None:
    m = load(R1, "sdpa")
    r1_mags = layer_magnitudes(m, batches["web"][worst_seq:worst_seq + 1])
    del m
    torch.cuda.empty_cache()

# ---- report ---------------------------------------------------------------
print("=" * 90)
print("1. VALIDITY  (must reproduce test_r1_ce_math.py: web teacher 2.2963, R1 2.4559)")
print(f"   teacher {ce[('teacher/sdpa', 'web')].mean():.4f}   R1 {ce[('r1/sdpa', 'web')].mean():.4f}")

for c in batches:
    print("\n" + "=" * 90)
    print(f"2. PER-SEQUENCE, {c}:  R5 - R1, paired over {N_SEQ} sequences")
    for impl in IMPLS:
        d = ce[(f"r5/{impl}", c)].mean(1) - ce[(f"r1/{impl}", c)].mean(1)
        srt = torch.sort(d, descending=True)
        print(f"   {impl:18} mean {d.mean():+.4f}  median {d.median():+.4f}  sd {d.std():.3f}  "
              f"#>+0.5 nat {int((d > 0.5).sum())}  #>+2 nat {int((d > 2).sum())}")
        print(f"   {'':18} worst five: " + "  ".join(
            f"seq{int(i)}:{v:+.2f}" for v, i in zip(srt.values[:5], srt.indices[:5])))
        trim = srt.values[5:]
        print(f"   {'':18} mean WITHOUT the worst five: {trim.mean():+.4f}")

    print(f"\n3. PER-POSITION, {c} (sdpa): mean CE in {NBIN} bins of {SEQ // NBIN} tokens")
    edges = [round(k * (SEQ - 1) / NBIN) for k in range(NBIN + 1)]
    hdr = "".join(f"{e0:>6}-{e1:<5}" for e0, e1 in zip(edges[:-1], edges[1:]))
    print(f"   {'':12}{hdr}")
    for who in ("teacher/sdpa", "r1/sdpa", "r5/sdpa"):
        t = ce[(who, c)]
        print(f"   {who:12}" + "".join(f"{t[:, e0:e1].mean():12.3f}" for e0, e1 in zip(edges[:-1], edges[1:])))
    dd = ce[("r5/sdpa", c)] - ce[("r1/sdpa", c)]
    print(f"   {'R5 - R1':12}" + "".join(f"{dd[:, e0:e1].mean():+12.3f}" for e0, e1 in zip(edges[:-1], edges[1:])))
    print("   -> excess GROWING with position = something accumulates (a recurrent")
    print("      state that stopped forgetting); FLAT = a uniform per-token penalty.")

print("\n" + "=" * 90)
print(f"4. HIDDEN-STATE MAGNITUDE on web seq{worst_seq} (R5's worst), sdpa:  max|h| and RMS per layer")
if r1_mags and r5_mags:
    print(f"   {'layer':>5}{'R1 max':>11}{'R5 max':>11}{'R1 rms':>9}{'R5 rms':>9}{'R5/R1 rms':>11}")
    for L, (a, b) in enumerate(zip(r1_mags, r5_mags)):
        flag = "  <--" if b[1] > 2 * a[1] else ""
        print(f"   {L:5}{a[0]:11.1f}{b[0]:11.1f}{a[1]:9.2f}{b[1]:9.2f}{b[1] / a[1]:11.2f}{flag}")
seq = batches["web"][worst_seq]
p = int((ce[("r5/sdpa", "web")][worst_seq] - ce[("r1/sdpa", "web")][worst_seq]).argmax())
print(f"   worst position {p}; text around it:")
print("   " + repr(tok.decode(seq[max(0, p - 40):p + 10]))[:300])

# ---- weight drift, CPU ----------------------------------------------------
print("\n" + "=" * 90)
print("5. WEIGHT DRIFT  R5 vs R1  (safetensors, CPU)")
from safetensors import safe_open


def tensors(d):
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise SystemExit(f"no safetensors in {d}")
    for f in files:
        with safe_open(f, "pt") as h:
            for k in h.keys():
                yield k, f


idx1 = dict(tensors(R1))
idx5 = dict(tensors(R5))
rows, same = [], 0
for k in sorted(set(idx1) & set(idx5)):
    with safe_open(idx1[k], "pt") as h:
        a = h.get_tensor(k).float()
    with safe_open(idx5[k], "pt") as h:
        b = h.get_tensor(k).float()
    if torch.equal(a, b):
        same += 1
        continue
    rows.append((k, b.norm().item() / max(a.norm().item(), 1e-12),
                 a.abs().max().item(), b.abs().max().item()))
print(f"   {same} tensors bit-identical (frozen: expect 147);  {len(rows)} differ;  "
      f"only-in-R1 {len(set(idx1) - set(idx5))}, only-in-R5 {len(set(idx5) - set(idx1))}")
print(f"   top 15 by norm ratio R5/R1:")
for k, r, ma, mb in sorted(rows, key=lambda x: -x[1])[:15]:
    print(f"     {r:7.2f}x   max|w| {ma:9.3f} -> {mb:9.3f}   {k}")
print(f"   top 10 by max|w| in R5:")
for k, r, ma, mb in sorted(rows, key=lambda x: -x[3])[:10]:
    print(f"     max|w| {ma:9.3f} -> {mb:9.3f}   ({r:.2f}x norm)   {k}")
print("\nDONE")