#!/usr/bin/env python
"""Inference speed: PREFILL (forward only) and DECODE, teacher vs R1-R4.   (v1.2, 2026-10-21)
 
    python speed_bench.py --smoke                 # 1-2 min per model: 3 lengths, both parts, the checks
    python speed_bench.py                         # the full measurement, resumable
    MODELS=teacher:r3_kda_only python speed_bench.py   # a subset (':' or ',')
 
WHAT IS MEASURED (thesis "inference analysis", protocol after Shaj et al. 2026 App. E.3)
  prefill   batch 1, one forward pass over T tokens, no cache (= E.3's "forward-only pass,
            test-time prompt processing").  Median of REPS timed passes after WARMUP passes,
            CUDA events.  Peak memory above the weights.
  decode    batch 1 and 8.  Build the cache for T tokens (chunked prefill, CHUNK tokens per
            call, use_cache=True), then generate greedily: 4 warm-up steps, 32 timed steps,
            median ms per step (= time per output token).  Cache size = memory still
            allocated after the prefill.  OOM is recorded as a result, not an error.
  T         128 ... 131,072 (powers of 2).  131,072 is beyond the teacher's configured 65,536
            positions: a COST measurement only -- no model's output quality is claimed at any
            length here (the students were trained at 2,048).
 
FAIRNESS -- identical for every model
  bf16, torch.no_grad, attn_implementation=flash_attention_2 (teacher and the students' 9
  attention layers), fla 0.5.2 chunk / fused_recurrent kernels for KDA/GDN (fla switches to
  fused_recurrent itself for <= 64 new tokens).  use_cache passed EXPLICITLY on every call --
  checkpoints store use_cache: false, and a flag set on one side only is a fabricated
  comparison (track_b.md §6).  logits only for the last position (logits_to_keep=1): the full
  128,004-wide output at 64k would be ~17 GB and is identical work for every model.  No
  attention_mask is passed (no padding exists): the students' mask path adds an unpad/repad
  copy in fla and a GPU sync in is_segment_mask, which a padding-free prompt never needs.
  Inputs: random token ids, BOS first, the same ids for every model (seeded by T and batch);
  runtime does not depend on content.  HF's default DynamicCache for the attention layers of
  every model (AFMKDACache subclasses it).  No torch.compile / CUDA graphs for anyone.
 
BUILT-IN CHECKS
  chunked prefill == one-shot prefill (T = 4096, last-token logits and the next decode step),
  teacher as the control for the bf16 floor -- decode at batch 8 relies on chunked prefill.
  drift: prefill at T = 8192 re-measured at the end of each model.
 
Results: $WORK_LUKAS/eval_results/speed/<model>.json, one record per point, written as it is
measured; an existing record is skipped, so a killed job resumes by resubmission.
"""
import argparse
import gc
import json
import os
import platform
import re
import statistics as st
import time
 
import torch
from transformers import AutoModelForCausalLM
 
W = os.environ.get("WORK_LUKAS", "/leonardo_work/EUHPC_D34_049/WORK_LUKAS")
O = f"{W}/output"
TEACHER = f"{W}/hf_cache/hub/models--arcee-ai--AFM-4.5B-Base/snapshots/dab7922f9c868d479b365410304f466b007b1c5a"
MODELS = {   # speed depends on the architecture only -> stage-1 checkpoints stand for every run
    "teacher": TEACHER,
    "r1_kda_hybrid": f"{O}/r1_stage1_hybrid31/checkpoint-4580",
    "r2_gdn_hybrid": f"{O}/r2_stage1_gdn_hybrid31/checkpoint-4580",
    "r3_kda_only": f"{O}/r3_stage1_kda_only/checkpoint-4580",
    "r4_gdn_only": f"{O}/r4_stage1_gdn_only/checkpoint-4580",
}
OUTDIR = f"{W}/eval_results/speed"
LENGTHS = [2 ** k for k in range(7, 18)]          # 128 ... 131072
BOS, VOCAB_LO, VOCAB_HI = 128000, 1000, 128000    # random ids avoid the special-token range
WARMUP, REPS, REPS_LONG = 3, 10, 5                # REPS_LONG for T >= 131072
DEC_WARM, DEC_STEPS = 4, 32
CHUNK = 4096                                      # tokens per row per prefill call when building a cache
DEV = "cuda"
 
 
# ---------------------------------------------------------------- helpers
def ids_for(T, B):
    g = torch.Generator().manual_seed(1_000_003 * T + B)
    x = torch.randint(VOCAB_LO, VOCAB_HI, (B, T), generator=g)
    x[:, 0] = BOS
    return x.to(DEV)
 
 
def mib(x):
    return round(x / 2 ** 20, 1)
 
 
def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
 
 
def clean():
    gc.collect()
    torch.cuda.empty_cache()
 
 
def load(path):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True,
                                             attn_implementation="flash_attention_2")
    return m.to(DEV).eval()
 
 
@torch.no_grad()
def prefill_once(model, x):
    return model(input_ids=x, use_cache=False, logits_to_keep=1).logits
 
 
@torch.no_grad()
def build_cache(model, x, chunk=CHUNK):
    """Chunked prefill with the cache on.  Returns (cache, last logits, seconds)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cache, out = None, None
    for s in range(0, x.shape[1], chunk):
        out = model(input_ids=x[:, s:s + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
    torch.cuda.synchronize()
    return cache, out.logits[:, -1].float(), time.perf_counter() - t0
 
 
@torch.no_grad()
def decode_step(model, cache, tok):
    out = model(input_ids=tok, past_key_values=cache, use_cache=True, logits_to_keep=1)
    return out.logits[:, -1].argmax(-1, keepdim=True), out.logits[:, -1]
 
 
# ---------------------------------------------------------------- the two measurements
def measure_prefill(model, T, base_alloc):
    x = ids_for(T, 1)
    for _ in range(WARMUP):
        prefill_once(model, x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(REPS_LONG if T >= 131072 else REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        prefill_once(model, x)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    med = st.median(ts)
    return dict(status="ok", ms=med, ms_min=min(ts), ms_max=max(ts), reps=len(ts),
                tok_per_s=T / (med / 1e3),
                peak_above_weights_mib=mib(torch.cuda.max_memory_allocated() - base_alloc))
 
 
def measure_decode(model, T, B):
    x = ids_for(T, B)
    clean()
    before = torch.cuda.memory_allocated()
    # v1.2: above batch 8 the chunk shrinks so each call still holds 8 x 4096 = 32k tokens --
    # otherwise an OOM at batch 32/64 would come from MLP activations of the prefill call, not
    # from the cache, and be misread.  Batch <= 8 is unchanged (4096 tokens per row per call).
    chunk = CHUNK if B <= 8 else max(256, CHUNK * 8 // B)
    try:
        cache, logits, t_pre = build_cache(model, x, chunk=chunk)
    except RuntimeError as e:                   # tag WHERE it failed; OOM detection reads the text
        raise RuntimeError(f"[during cache build, chunk {chunk}] {e}") from None
    torch.cuda.synchronize()
    cache_bytes = torch.cuda.memory_allocated() - before      # what the cache keeps alive
    tok = logits.argmax(-1, keepdim=True)
    for _ in range(DEC_WARM):
        tok, _ = decode_step(model, cache, tok)
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(DEC_STEPS)]
    torch.cuda.synchronize()
    w0 = time.perf_counter()
    for a, b in ev:
        a.record()
        tok, _ = decode_step(model, cache, tok)
        b.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - w0) * 1e3 / DEC_STEPS     # includes all host-side gaps
    ts = [a.elapsed_time(b) for a, b in ev]
    del cache
    clean()
    med = st.median(ts)
    return dict(status="ok", ms_per_token=med, ms_min=min(ts), ms_max=max(ts), steps=len(ts),
                wall_ms_per_token=round(wall_ms, 3), tok_per_s=B / (med / 1e3), cache_mib=mib(cache_bytes),
                prefill_with_cache_s=round(t_pre, 3))
 
 
@torch.no_grad()
def chunk_check(model, T=4096, small=1024):
    """Chunked prefill (small chunks) vs one-shot prefill, both with the cache on: last-token logits
    and the logits of the next decode step.  The teacher's numbers are the bf16 floor.
    v1.1: BOTH paths are fed the SAME next token (one-shot's argmax).  v1 fed each path its own
    argmax, so a near-tie at the last prompt position (random-token prompts give flat
    distributions) made the step comparison compare two different inputs -- R1's smoke 'failure'."""
    x = ids_for(T, 1)
    c1, l1, _ = build_cache(model, x, chunk=T)
    nxt = l1.argmax(-1, keepdim=True)
    t1, n1 = decode_step(model, c1, nxt)
    del c1
    clean()
    c2, l2, _ = build_cache(model, x, chunk=small)
    t2, n2 = decode_step(model, c2, nxt)
    del c2
    clean()
    return dict(T=T, chunk=small,
                prefill_logit_maxdiff=float((l1 - l2).abs().max()), prefill_argmax_same=bool((l1.argmax(-1) == l2.argmax(-1)).all()),
                step_logit_maxdiff=float((n1.float() - n2.float()).abs().max()), step_argmax_same=bool((t1 == t2).all()),
                logit_scale=float(l1.abs().max()),
                prefill_top2_gap=float((lambda v: v[..., 0] - v[..., 1])(l1.topk(2, -1).values).min()))
 
 
# ---------------------------------------------------------------- bookkeeping
def save(path, res):
    tmp = path + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, path)
 
 
def provenance():
    from importlib.metadata import version
    v = {}
    for p in ("torch", "transformers", "flash-linear-attention", "flash_attn"):
        try:
            v[p] = version(p)
        except Exception:
            v[p] = None
    return dict(gpu=torch.cuda.get_device_name(0), host=platform.node(), versions=v,
                slurm_job=os.environ.get("SLURM_JOB_ID"), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                date=time.strftime("%Y-%m-%d %H:%M:%S"))
 
 
def run_model(name, path, lengths, batches, outdir, smoke):
    f = f"{outdir}/{name}{'.smoke' if smoke else ''}.json"
    res = json.load(open(f)) if os.path.exists(f) else {"name": name, "path": path, "points": {}}
    pts = res["points"]
    todo = [f"prefill|b1|T{T}" for T in lengths] + [f"decode|b{B}|T{T}" for B in batches for T in lengths]
    todo += ["chunk_check", "recheck|prefill|b1|T8192"]
    if "chunk_check" in pts and "prefill_top2_gap" not in pts["chunk_check"]:
        del pts["chunk_check"]                   # a v1 check (different-token step) -> redo it
    if all(k in pts for k in todo):
        print(f"=== {name}: complete, skipped", flush=True)
        return
    print(f"\n=== {name}   {path}", flush=True)
    t0 = time.time()
    model = load(path)
    clean()
    base = torch.cuda.memory_allocated()
    res.update(params=sum(p.numel() for p in model.parameters()), weights_mib=mib(base),
               model_class=type(model).__name__, provenance=provenance(),
               attn_impl=model.config._attn_implementation)
    print(f"    loaded {res['model_class']} in {time.time() - t0:.0f}s   params {res['params']:,}   "
          f"weights {res['weights_mib']} MiB   attn {res['attn_impl']}", flush=True)
 
    def point(key, fn, label):
        if key in pts:
            return pts[key]["status"]
        try:
            r = fn()
        except Exception as e:                  # an OOM is a result; anything else is recorded and shown
            r = dict(status="oom" if is_oom(e) else "error", error=f"{type(e).__name__}: {str(e)[:300]}")
        clean()                                 # outside the except: the traceback no longer pins tensors
        pts[key] = r
        save(f, res)
        print(f"    {label:22} " + ("  ".join(f"{k} {v}" for k, v in r.items() if k != "status")
                                     if r["status"] == "ok" else r["status"].upper() + "  " + r.get("error", "")), flush=True)
        return r["status"]
 
    stop = False
    for T in lengths:                               # prefill: once a length OOMs, longer ones will too
        if stop:
            pts.setdefault(f"prefill|b1|T{T}", dict(status="oom", error="skipped after a shorter OOM"))
            continue
        stop = point(f"prefill|b1|T{T}", lambda: measure_prefill(model, T, base), f"prefill T={T}") == "oom"
    for B in batches:
        stop = False
        for T in lengths:
            if stop:
                pts.setdefault(f"decode|b{B}|T{T}", dict(status="oom", error="skipped after a shorter OOM"))
                continue
            stop = point(f"decode|b{B}|T{T}", lambda: measure_decode(model, T, B), f"decode B={B} T={T}") == "oom"
    point("chunk_check", lambda: dict(status="ok", **chunk_check(model)), "chunked-prefill check")
    point("recheck|prefill|b1|T8192", lambda: measure_prefill(model, 8192, base), "drift recheck T=8192")
    save(f, res)
    del model
    clean()
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 lengths, batch 1 and 8, own result files")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    # ':' or ',' separated -- use ':' through sbatch --export, which splits on commas
    names = [n for n in re.split(r"[,:]", os.environ.get("MODELS", ":".join(MODELS))) if n]
    bad = [n for n in names if n not in MODELS or not os.path.isfile(f"{MODELS.get(n, '')}/config.json")]
    if bad:
        raise SystemExit(f"FATAL: unknown model or no config.json: {bad}")
    lengths = [512, 4096, 16384] if a.smoke else LENGTHS
    # DEC_BATCHES=1:8:32:64 adds decode batch sizes; existing points are skipped, so a later job
    # with extra batches measures only the new ones
    batches = tuple(int(b) for b in re.split(r"[,:]", os.environ.get("DEC_BATCHES", "1:8")) if b)
    print(f"lengths {lengths}   decode batches {batches}   -> {OUTDIR}{'  (SMOKE)' if a.smoke else ''}", flush=True)
    for n in names:
        run_model(n, MODELS[n], lengths, batches, OUTDIR, a.smoke)
    print("\nDONE", flush=True)
 
 
if __name__ == "__main__":
    main()
