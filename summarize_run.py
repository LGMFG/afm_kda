#!/usr/bin/env python
"""Condense DistillKit training logs into one screen -- stage 1 AND stage 2.

    python summarize_run.py RUN.out [RUN.err]               one run, full report
    python summarize_run.py --compare A.out B.out C.out ...  several runs side by side
    (the .err is found automatically when it sits next to the .out)

v3.1 (2026-10-16): tokens per eval from the epoch, spikes vs a running median, loss-last-20 in --compare.
v3 (2026-10-15).  What changed against v2, and why:
  * The loss components are DETECTED from the log (every key that starts with
    `distillation_loss/`).  v2 hard-coded the stage-1 trio (CE / KL / hs_cosine),
    so on a stage-2 log -- which logs only `distillation_loss/1_kl` -- the whole
    component section silently disappeared.
  * The stage is detected (hs_cosine logged -> stage 1, KL only -> stage 2) and
    every reference value is stage-specific.  v2 compared a stage-2 eval_loss
    with R1's stage-1 composite 0.5686 ("final vs R1 -0.2955"): that number is
    MEANINGLESS -- stage-1 eval_loss is 0.2 CE + 0.2 KL + 0.6 hs_cosine, stage-2
    eval_loss is KL alone.  Both are on the SAME 200 held-out documents, but
    they are different quantities.
  * `Froze N tensors` is counted per N and checked against the stage
    (stage 1: 147 x 16 ranks, stage 2: none).
  * A seed check: a seed replicate must NOT start with R1's / R2's first
    microbatch (pre-registered in make_seed_replicates.py).
  * Tokens per eval point are computed from the step count (eval_steps from
    --eval-steps, or inferred), so stage-2 curves read in tokens.
  * --compare prints the eval curves of several runs in columns, plus one
    summary row each.  Only compare runs of the SAME stage (same eval quantity).

Extraction method (unchanged, = the run log's):
  start = FIRST logged microbatch, end = MEAN of the LAST TWENTY microbatches.
"""
import argparse
import ast
import os
import re
import statistics as st

DICT = re.compile(r"\{'[^{}]*\}")
TOK_PER_STEP = 32 * 2048            # global batch 32 x 2048 -- constant for every run here

# ---- references (first microbatch -> mean of last 20; final eval_loss on the 200 held-out docs)
REF_S1 = {
    "R1": dict(start=(7.986, 9.286, 0.844), end=(2.261, 0.314, 0.069), eval=0.5686),
    "R2": dict(start=(7.981, 9.349, 0.840), end=(2.283, 0.355, 0.070), eval=0.5814),
    "R3": dict(start=(8.590, 10.040, 0.837), end=(2.276, 0.334, 0.068), eval=0.5744),
    "R4": dict(start=(8.531, 9.914, 0.846), end=(2.303, 0.379, 0.075), eval=0.5937),
}
# Stage-2 reference: R1-S2 (job 59749921), const 1e-5.  eval_loss = KL (T=2) on the same 200 docs.
REF_S2 = {"R1-S2": dict(eval_first=0.3379, eval_final=0.2731, train_end=0.2764)}


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
    if not path or not os.path.exists(path):
        return None
    rx = re.compile(pat)
    return sum(len(rx.findall(line)) for line in open(path, errors="replace"))


def grab(path, pats, limit=14):
    if not path or not os.path.exists(path):
        return []
    rx = re.compile(pats)
    seen, out = set(), []
    for line in open(path, errors="replace"):
        line = line.rstrip()
        if rx.search(line) and line not in seen:
            seen.add(line)
            out.append(line[:160])
    return out[:limit]


def short_comp(k):
    """'distillation_loss/2_kl' -> 'KL'"""
    n = k.split("/", 1)[1].split("_", 1)[-1]
    return {"cross_entropy": "CE", "kl": "KL", "hs_cosine": "hs_cos"}.get(n, n[:8])


class Run:
    def __init__(self, out, err=None, eval_steps=None, max_steps=None):
        self.out, self.err = out, err
        self.name = re.sub(r"-\d+\.out$", "", os.path.basename(out))
        d = parse(out)
        self.comp_keys = sorted({k for x in d for k in x if str(k).startswith("distillation_loss/")})
        self.dl = [x for x in d if any(k in x for k in self.comp_keys)]
        self.tl = [x for x in d if "learning_rate" in x]
        self.ev = [x for x in d if "eval_loss" in x]
        self.fin = [x for x in d if "train_runtime" in x]
        comps = [short_comp(k) for k in self.comp_keys]
        self.stage = 1 if any(c.startswith("hs") for c in comps) else (2 if comps else None)
        # steps: logging_steps is 10 in every config of this project
        self.max_steps = max_steps or (10 * len(self.tl) if self.tl else None)
        n_ev = len(self.ev)
        self.eval_steps = eval_steps or (self.max_steps // n_ev if (self.max_steps and n_ev) else None)

    # first microbatch / mean of last 20 of each component
    def components(self):
        if not self.dl:
            return None, None
        first, last = [], []
        for k in self.comp_keys:
            xs = [x[k] for x in self.dl if isinstance(x.get(k), float)]
            first.append(xs[0] if xs else float("nan"))
            last.append(st.mean(xs[-20:]) if xs else float("nan"))
        return first, last

    def eval_tokens(self, i):
        """Tokens at eval point i, from its EPOCH (exact for the last, irregular eval too).
        v3.0 used (i+1) x eval_steps, which mislabelled stage 1: eval_steps there is 500,
        not max_steps/10, and the 10th eval is the end-of-run eval at step 4580."""
        e = self.ev[i].get("epoch") if i < len(self.ev) else None
        e_end = self.fin[-1].get("epoch") if self.fin else (self.ev[-1].get("epoch") if self.ev else None)
        if isinstance(e, float) and isinstance(e_end, float) and e_end > 0 and self.max_steps:
            return e / e_end * self.max_steps * TOK_PER_STEP
        return (i + 1) * self.eval_steps * TOK_PER_STEP if self.eval_steps else None

    def loss_last20(self):
        ls = [x["loss"] for x in self.tl if isinstance(x.get("loss"), float)]
        return st.mean(ls[-20:]) if ls else float("nan")

    def spikes(self):
        """grad_norm spikes AFTER warm-up: > 5x the running median of the previous 50 logs.
        v3.0 compared against the whole-run median, which counted the (normal) large
        gradients of the first ~100 stage-1 steps as spikes."""
        g = self.gnorm()
        skip = max(10, len(g) // 20)                       # first 5 % of logs = warm-up
        out = []
        for j in range(skip, len(g)):
            ref = st.median(x for _, x in g[max(0, j - 50):j])
            if g[j][1] > 5 * ref:
                out.append((g[j][0], g[j][1], ref))
        return out

    def gnorm(self):
        g = [(x.get("epoch"), x["grad_norm"]) for x in self.tl if isinstance(x.get("grad_norm"), float)]
        return g

    def throughput(self):
        if not self.fin:
            return None
        F = self.fin[-1]
        sps = F.get("train_samples_per_second")
        return sps * 2048 if sps else None


def report(r, args):
    print(f"######## {r.name}   stage {r.stage or '?'}   components: "
          f"{', '.join(short_comp(k) for k in r.comp_keys) or 'none found'}")

    print("\n=== provenance / guards")
    for line in grab(r.out, r"guards OK|output_path guard|^VERSIONS|PREWARM|FATAL|FAILED|DONE: stamped|"
                            r"^[0-9a-f]{7,} |\"dirty\"|build_commit|git_commit|start global_step"):
        print("   ", line)

    print("\n=== health")
    froze = {}
    if os.path.exists(r.out):
        for line in open(r.out, errors="replace"):
            for n in re.findall(r"Froze (\d+) tensors", line):
                froze[n] = froze.get(n, 0) + 1
    froze_txt = ", ".join(f"{n} x{c}" for n, c in froze.items()) or "none"
    froze_want = "147 x16" if r.stage == 1 else "none (nothing frozen)" if r.stage == 2 else "?"
    rows = [
        ("Froze N tensors", froze_txt, froze_want),
        ("train_runtime dicts", len(r.fin), "1 (2+ = the job split)"),
        ("train-log dicts", len(r.tl), f"max_steps/10 = {r.max_steps // 10}" if args.max_steps else "max_steps/10"),
        ("eval dicts", len(r.ev), "10 (R5: 20)"),
        ("Tokenizing train dataset (err)", count(r.err, r"Tokenizing train dataset"), "0 = cache hit"),
        ("Packing train dataset (err)", count(r.err, r"Packing train dataset"), "0"),
        ("roll back to CPU", (count(r.out, r"roll back to CPU") or 0) + (count(r.err, r"roll back to CPU") or 0), "0"),
        ("Traceback (err)", count(r.err, r"Traceback"), "0"),
        ("NCCL watchdog / timeout (err)", count(r.err, r"Watchdog caught|timed out|DistBackendError"), "0"),
        ("OutOfMemory", (count(r.out, r"OutOfMemory|out of memory") or 0) + (count(r.err, r"OutOfMemory|out of memory") or 0), "0"),
    ]
    g = r.gnorm()
    if g:
        med = st.median(x for _, x in g)
        mx = max(g, key=lambda x: x[1])
        rows.append(("max grad_norm", f"{mx[1]:.2f} @ ep {mx[0]:.3f}", "< ~20; stage 2 runs above the 0.5 clip"))
        sp = r.spikes()
        rows.append(("grad_norm spikes after warm-up", len(sp), "few; each listed below (>5x running median)"))
    ups = sum(1 for x, y in zip(r.ev, r.ev[1:]) if y["eval_loss"] > x["eval_loss"] + 0.01)
    rows.append(("eval_loss rises > 0.01", ups, "0"))
    nan = sum(1 for x in r.dl + r.tl + r.ev for v in x.values()
              if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))))
    rows.append(("NaN/inf in logged values", nan, "0"))
    for name, v, want in rows:
        print(f"    {name:32} {'n/a' if v is None else v!s:>16}   expect {want}")
    for e, v, ref in (r.spikes() if g else [])[:10]:
        print(f"        spike: grad_norm {v:.2f} at epoch {e:.4f} (running median {ref:.3f}, {v / ref:.0f}x)")

    if r.tl:
        print(f"\n=== schedule  ({len(r.tl)} train-log dicts)")
        lrs = [(x.get("epoch"), x["learning_rate"]) for x in r.tl]
        pk = max(lrs, key=lambda x: x[1])
        print(f"    lr      first {lrs[0][1]:.3e}   peak {pk[1]:.3e} @ epoch {pk[0]:.4f}   last {lrs[-1][1]:.3e}")
        if g:
            print(f"    gnorm   first {g[0][1]:.3f}   mean last 20 {st.mean(x for _, x in g[-20:]):.4f}")
        ls = [x["loss"] for x in r.tl if isinstance(x.get("loss"), float)]
        if ls:
            what = "= KL in stage 2" if r.stage == 2 else "= weighted sum"
            print(f"    loss    first {ls[0]:.4f}   mean last 20 {st.mean(ls[-20:]):.4f}   ({what})")

    first, last = r.components()
    if first:
        names = [short_comp(k) for k in r.comp_keys]
        print(f"\n=== distillation components  ({len(r.dl)} rank-0 microbatches)")
        print(f"    {'':22}" + "".join(f"{n:>9}" for n in names))
        print(f"    {'first microbatch':22}" + "".join(f"{x:9.3f}" for x in first))
        print(f"    {'mean of last 20':22}" + "".join(f"{x:9.3f}" for x in last))
        if r.stage == 1:
            for k in ("R1", "R2"):
                print(f"    {k + ' first (ref)':22}" + "".join(f"{x:9.3f}" for x in REF_S1[k]["start"]))
            clash = [k for k, v in REF_S1.items() if all(abs(a - b) < 5e-4 for a, b in zip(first, v["start"]))]
            print("    SEED CHECK: " + (f"!! first microbatch IDENTICAL to {clash} -- the seed did not take effect"
                                         if clash else "first microbatch differs from R1-R4 -> new data order (ok)"))
        bins = {}
        for x in r.dl:
            e = x.get("epoch") or 0.0
            bins.setdefault(round(e // 0.1 * 0.1, 1), []).append(x)
        print(f"\n    {'epoch':>7}{'n':>6}" + "".join(f"{n:>9}" for n in names))
        for b in sorted(bins):
            B = bins[b]
            print(f"    {b:7.1f}{len(B):6}" + "".join(
                f"{st.mean(x[k] for x in B if isinstance(x.get(k), float)):9.3f}" for k in r.comp_keys))

    if r.ev:
        what = ("composite 0.2 CE + 0.2 KL + 0.6 hs_cos" if r.stage == 1
                else "KL (T=2) only" if r.stage == 2 else "?")
        print(f"\n=== held-out eval_loss = {what}, on the SAME 200 held-out docs as every run")
        prev = None
        for i, x in enumerate(r.ev):
            t = r.eval_tokens(i)
            ts = f"{t / 1e6:7.0f} M" if t else "      -"
            delta = "" if prev is None else f"{x['eval_loss'] - prev:+.4f}"
            print(f"    {i + 1:3}  epoch {x.get('epoch', float('nan')):.4f}  {ts}   eval_loss {x['eval_loss']:.4f}  {delta}")
            prev = x["eval_loss"]
        fe = r.ev[-1]["eval_loss"]
        if r.stage == 1:
            print("    final vs stage-1 refs: " + "  ".join(f"{k} {fe - v['eval']:+.4f}" for k, v in REF_S1.items()))
        elif r.stage == 2:
            ref = REF_S2["R1-S2"]
            print(f"    first -> final: {r.ev[0]['eval_loss']:.4f} -> {fe:.4f} "
                  f"({(fe / r.ev[0]['eval_loss'] - 1) * 100:+.1f} %)")
            print(f"    vs R1-S2 final {ref['eval_final']}: {fe - ref['eval_final']:+.4f}   "
                  "(comparable ONLY with other stage-2 runs; NOT with 0.5686)")

    if r.fin:
        F = r.fin[-1]
        rt, sps = F.get("train_runtime"), F.get("train_samples_per_second")
        print("\n=== final")
        print(f"    train_runtime {rt:.0f} s = {rt / 3600:.2f} h   epoch {F.get('epoch')}")
        if sps:
            seqs = sps * rt
            print(f"    throughput    {sps * 2048:,.0f} tok/s   ({sps:.3f} samples/s)")
            print(f"    tokens, measured   samples/s x runtime x 2048 = {seqs * 2048 / 1e9:.4f} B ({seqs:,.0f} sequences)")
        if r.max_steps:
            print(f"    tokens, configured {r.max_steps} steps x 65,536 = {r.max_steps * TOK_PER_STEP / 1e9:.4f} B"
                  f"{'' if args.max_steps else '   (max_steps inferred from 10 x train-log dicts)'}")
        for k in ("train_loss", "train_mem_gpu_alloc_delta", "train_mem_gpu_peaked_delta"):
            if k in F:
                v = F[k]
                print(f"    {k:28} {v / 2**30:.2f} GiB" if "mem" in k else f"    {k:28} {v}")
    print()


def compare(runs):
    stages = {r.stage for r in runs}
    if len(stages) > 1:
        print(f"!! mixing stages {sorted(s for s in stages if s)}: stage-1 eval_loss is a composite, "
              "stage-2 eval_loss is KL -- the columns are NOT comparable.\n")
    w = max(12, max(len(r.name) for r in runs[:1]))
    names = [r.name[-22:] for r in runs]
    print("=== held-out eval_loss by eval point (same 200 docs for every run; tokens from the first run's epochs)")
    print(f"    {'#':>3} {'tokens':>8}  " + "".join(f"{n:>24}" for n in names))
    n = max(len(r.ev) for r in runs)
    for i in range(n):
        t = runs[0].eval_tokens(i)
        ts = f"{t / 1e6:6.0f}M" if t else "     -"
        cells = [f"{r.ev[i]['eval_loss']:24.4f}" if i < len(r.ev) else f"{'':>24}" for r in runs]
        print(f"    {i + 1:3} {ts:>8}  " + "".join(cells))
    print("\n=== summary")
    print(f"    {'run':34}{'stage':>6}{'hours':>7}{'tok/s':>9}{'eval first':>11}{'eval final':>11}"
          f"{'max gnorm':>10}{'spikes':>7}{'rises':>6}{'loss last20':>12}  components, last 20 rank-0 microbatches")
    for r in runs:
        F = r.fin[-1] if r.fin else {}
        hrs = F.get("train_runtime", float("nan")) / 3600
        tps = r.throughput() or float("nan")
        g = r.gnorm()
        mg = max(x for _, x in g) if g else float("nan")
        ups = sum(1 for x, y in zip(r.ev, r.ev[1:]) if y["eval_loss"] > x["eval_loss"] + 0.01)
        _, last = r.components()
        comp = " / ".join(f"{short_comp(k)} {v:.3f}" for k, v in zip(r.comp_keys, last)) if last else "-"
        e0 = r.ev[0]["eval_loss"] if r.ev else float("nan")
        e1 = r.ev[-1]["eval_loss"] if r.ev else float("nan")
        print(f"    {r.name[-34:]:34}{r.stage or '?':>6}{hrs:7.2f}{tps:9,.0f}{e0:11.4f}{e1:11.4f}{mg:10.2f}"
              f"{len(r.spikes()):7}{ups:6}{r.loss_last20():12.4f}  {comp}")
    print("\n    'loss last20' = mean of the last 20 logged `loss` values (200 steps, averaged over ALL ranks):")
    print("    the stable train-side number.  The component column is 20 rank-0 microbatches only -- noisy.")
    if 1 in stages:
        print("\n    stage-1 refs (final eval_loss): " + "  ".join(f"{k} {v['eval']}" for k, v in REF_S1.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="RUN.out [RUN.err]   or, with --compare, several .out files")
    ap.add_argument("--compare", action="store_true", help="put several runs side by side")
    ap.add_argument("--max-steps", type=int, default=None, help="default: inferred as 10 x train-log dicts")
    ap.add_argument("--eval-steps", type=int, default=None, help="default: max_steps / number of evals")
    a = ap.parse_args()

    if a.compare:
        outs = [x for x in a.logs if not x.endswith(".err")]
        compare([Run(o, None, a.eval_steps, a.max_steps) for o in outs])
        return
    out = a.logs[0]
    err = a.logs[1] if len(a.logs) > 1 else (
        out[:-4] + ".err" if out.endswith(".out") and os.path.exists(out[:-4] + ".err") else None)
    report(Run(out, err, a.eval_steps, a.max_steps), a)


if __name__ == "__main__":
    main()