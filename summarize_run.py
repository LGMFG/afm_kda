#!/usr/bin/env python
"""Condense a DistillKit training log into one screen.

    python summarize_run.py RUN.out [RUN.err] [--epoch-tokens N] [--max-steps N]

Parses every Python-dict line the Trainer prints (values may be quoted
strings or numbers, and tqdm junk may share the line), and reports exactly
the quantities this project's run log records -- with the SAME extraction
method used for R1-R4:
  * start column = the FIRST logged distillation microbatch
  * end column   = the MEAN of the LAST TWENTY
plus the held-out eval curve, the LR schedule, the token count two ways,
and the health checks from the pre-registered tables.

--epoch-tokens converts `epoch` into tokens.  For R5 it is the measured
one-epoch size, 1,118,292 x 2048 = 2,290,262,016.  For R1-R4 it is NOT the
corpus -- `epoch` there is relative to num_samples=500000.
"""
import argparse
import ast
import re
import statistics as st

R1_REF = dict(start=(7.986, 9.286, 0.844), end=(2.261, 0.314, 0.069),
              eval_final=0.5686, toks=300.15e6)
DICT = re.compile(r"\{'[^{}]*\}")
KEYS = ("distillation_loss/1_cross_entropy", "distillation_loss/2_kl",
        "distillation_loss/3_hs_cosine")


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse(path):
    out = []
    for line in open(path, errors="replace"):
        for m in DICT.finditer(line):
            try:
                d = ast.literal_eval(m.group(0))
            except Exception:
                continue
            if isinstance(d, dict):
                out.append({k: (f(v) if f(v) is not None else v) for k, v in d.items()})
    return out


def count(path, pat):
    if not path:
        return None
    rx = re.compile(pat)
    return sum(len(rx.findall(line)) for line in open(path, errors="replace"))


def grab(path, pats, limit=14):
    if not path:
        return []
    rx = re.compile(pats)
    seen, out = set(), []
    for line in open(path, errors="replace"):
        line = line.rstrip()
        if rx.search(line) and line not in seen:
            seen.add(line)
            out.append(line[:160])
    return out[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("err", nargs="?")
    ap.add_argument("--epoch-tokens", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--global-batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=2048)
    a = ap.parse_args()

    dicts = parse(a.out)
    dl = [d for d in dicts if KEYS[0] in d]
    tl = [d for d in dicts if "learning_rate" in d]
    ev = [d for d in dicts if "eval_loss" in d]
    fin = [d for d in dicts if "train_runtime" in d]
    tok = (lambda e: e * a.epoch_tokens) if a.epoch_tokens else None

    print("=== provenance / guards")
    for line in grab(a.out, r"output_path guard|^VERSIONS|PREWARM|FATAL|FAILED|DONE: stamped|"
                            r"^[0-9a-f]{7,} |\"dirty\"|build_commit|git_commit"):
        print("   ", line)

    print("\n=== health")
    rows = [
        ("Froze 147 tensors",            count(a.out, r"Froze 147 tensors"), "16"),
        ("train_runtime dicts",          len(fin), "1"),
        ("'loss' (train-log) dicts",     len(tl), f"{a.max_steps // 10}" if a.max_steps else "max_steps/10"),
        ("eval dicts",                   len(ev), "20 for R5"),
        ("Tokenizing train dataset",     count(a.err, r"Tokenizing train dataset"), "0 = cache hit"),
        ("Packing train dataset",        count(a.err, r"Packing train dataset"), "0"),
        ("roll back to CPU (out+err)",   (count(a.out, r"roll back to CPU") or 0) + (count(a.err, r"roll back to CPU") or 0), "0"),
        ("Traceback (err)",              count(a.err, r"Traceback"), "0"),
        ("NCCL watchdog / timeout (err)", count(a.err, r"Watchdog caught|timed out|DistBackendError"), "0"),
        ("OutOfMemory (out+err)",        (count(a.out, r"OutOfMemory|out of memory") or 0) + (count(a.err, r"OutOfMemory|out of memory") or 0), "0"),
    ]
    nan = sum(1 for d in dl + tl + ev for v in d.values()
              if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))))
    rows.append(("NaN/inf in logged values", nan, "0"))
    for name, v, want in rows:
        print(f"    {name:32} {'n/a' if v is None else v:>8}   expect {want}")

    if tl:
        print(f"\n=== schedule  ({len(tl)} train-log dicts)")
        lrs = [(d.get("epoch"), d["learning_rate"]) for d in tl]
        pk = max(lrs, key=lambda x: x[1])
        print(f"    lr      first {lrs[0][1]:.3e}   peak {pk[1]:.3e} @ epoch {pk[0]:.4f}   last {lrs[-1][1]:.3e}")
        gn = [(d.get("epoch"), d.get("grad_norm")) for d in tl if d.get("grad_norm") is not None]
        if gn:
            mx = max(gn, key=lambda x: x[1])
            print(f"    gnorm   first {gn[0][1]:.3f}   max {mx[1]:.3f} @ epoch {mx[0]:.4f}   "
                  f"mean last 20 {st.mean(g for _, g in gn[-20:]):.4f}")
        ls = [d["loss"] for d in tl if "loss" in d]
        if ls:
            print(f"    loss    first {ls[0]:.4f}   mean last 20 {st.mean(ls[-20:]):.4f}")

    if dl:
        print(f"\n=== distillation components  ({len(dl)} rank-0 microbatches)")
        first = [dl[0][k] for k in KEYS]
        last = [st.mean(d[k] for d in dl[-20:]) for k in KEYS]
        print(f"    {'':22}{'CE':>8}{'KL':>8}{'hs_cos':>8}")
        print(f"    {'first microbatch':22}" + "".join(f"{x:8.3f}" for x in first))
        print(f"    {'mean of last 20':22}" + "".join(f"{x:8.3f}" for x in last))
        print(f"    {'R1 first (ref)':22}" + "".join(f"{x:8.3f}" for x in R1_REF['start']))
        print(f"    {'R1 last 20 (ref)':22}" + "".join(f"{x:8.3f}" for x in R1_REF['end']))
        # binned by epoch (0.05) -- and by tokens if known
        bins = {}
        for d in dl:
            e = d.get("epoch") or 0.0
            bins.setdefault(round(e // 0.05 * 0.05, 2), []).append(d)
        print(f"\n    {'epoch':>7}{'tokens':>9}{'n':>5}{'CE':>8}{'KL':>8}{'hs_cos':>8}")
        for b in sorted(bins):
            B = bins[b]
            t = f"{tok(b + 0.025) / 1e6:7.0f}M" if tok else "      -"
            print(f"    {b:7.2f}{t:>9}{len(B):5}"
                  + "".join(f"{st.mean(x[k] for x in B):8.3f}" for k in KEYS))

    if ev:
        print(f"\n=== held-out eval_loss  ({len(ev)} evals; R1 final {R1_REF['eval_final']} "
              "on the SAME 200 documents -- same seed, same num_eval_samples)")
        prev = None
        for i, d in enumerate(ev, 1):
            e = d.get("epoch")
            t = f"{tok(e) / 1e9:6.3f} B" if (tok and e is not None) else "     -"
            delta = "" if prev is None else f"{d['eval_loss'] - prev:+.4f}"
            print(f"    {i:3}  epoch {e if e is not None else float('nan'):.4f}  {t}   "
                  f"eval_loss {d['eval_loss']:.4f}  {delta}")
            prev = d["eval_loss"]
        print(f"    final vs R1: {ev[-1]['eval_loss'] - R1_REF['eval_final']:+.4f}")

    if fin:
        F = fin[-1]
        rt, sps = F.get("train_runtime"), F.get("train_samples_per_second")
        print("\n=== final")
        print(f"    train_runtime {rt:.0f} s = {rt / 3600:.2f} h   epoch {F.get('epoch')}")
        if sps:
            seqs = sps * rt
            print(f"    throughput    {sps * a.seq:,.0f} tok/s   ({sps:.3f} samples/s)")
            print(f"    tokens, measured   samples/s x runtime x {a.seq} = {seqs * a.seq / 1e9:.4f} B "
                  f"({seqs:,.0f} sequences)")
        if a.max_steps:
            cfg = a.max_steps * a.global_batch
            print(f"    tokens, configured max_steps x {a.global_batch} x {a.seq}  = {cfg * a.seq / 1e9:.4f} B "
                  f"({cfg:,} sequences)")
        for k in ("train_loss", "train_mem_gpu_alloc_delta", "train_mem_gpu_peaked_delta"):
            if k in F:
                v = F[k]
                print(f"    {k:28} {v / 2**30:.2f} GiB" if "mem" in k else f"    {k:28} {v}")


if __name__ == "__main__":
    main()