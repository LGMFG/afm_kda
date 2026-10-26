#!/usr/bin/env python
"""Arithmetic probe v2 -- the STAGE-2 / SEED / FINEMATH comparisons.   (v4, 2026-10-18)

    python arith_stage2.py                      # default: $WORK_LUKAS/eval_results/arith_probe_v2
    python arith_stage2.py <results dir>

arith_compare.py scores the stage-1 pre-registration (P0-P6) and only knows the
stage-1 model names.  This script reads the SAME result files (same probe, same
2,286 problems, byte-identical prompts) and adds the models queued by
queue_evals.sh:

  r1_s2_kda_hyb, r2_s2_gdn_hyb, r3_s2_kda_only, r4_s2_gdn_only   stage 2, const 1e-5, 1 B
  r1_s2cos_kda_hyb                                                stage 2, cosine 5e-5 -> 1e-5
  r1_s2_ck01526 ... r1_s2_ck13734                                 R1-S2 checkpoints (curve)
  r1_s1_kda_hyb_seed43/44, r2_s1_gdn_hyb_seed43/44                stage-1 seed replicates
  r1_s1math_kda_hyb, r1_s2math_kda_hyb, r1fm_s2math_kda_hyb        FineMath runs (v2, 2026-10-18)
  r1_s2math_ck01526 ... r1_s2math_ck13734                         R1-S2-math checkpoints (v3: second curve)
  r2_s2math_gdn_hyb, r3_s2math_kda_only, r4_s2math_gdn_only       FineMath stage 2 on R2-R4 (v4)

It prints (1) the accuracy table, (2) paired stage-2 minus stage-1 per student
with the PRE-REGISTERED check (rev 36/42: R1-S2 pooled >= R1 + 5 points, i.e.
>= 43.3 %), (3) KDA - GDN within stage 2, (4) cosine - constant, (5) the R1-S2
token curves (FineWeb and FineMath stage 2, side by side, with the paired difference at
equal tokens and the slope per 100 M), (6) the seed spread of the pooled score and of KDA - GDN.
All differences are paired over identical problems, bootstrapped within condition.
"""
import glob
import os
import statistics as st
import sys

from arith_compare import TEACHER, R1, R2, R3, R4, acc, load, paired_boot

S2 = {"r1_s2_kda_hyb": R1, "r2_s2_gdn_hyb": R2, "r3_s2_kda_only": R3, "r4_s2_gdn_only": R4}
COS = "r1_s2cos_kda_hyb"
CURVES = [  # (label, checkpoint prefix, final model = step 15260); both start from R1 (step 0)
    ("R1-S2  (FineWeb stage 2)", "r1_s2_ck", "r1_s2_kda_hyb"),
    ("R1-S2-math (FineMath stage 2)", "r1_s2math_ck", "r1_s2math_kda_hyb"),
]
CURVE_CONDS = ["add_3", "sub_3", "add_4", "sub_4", "mul_2"]   # printed per checkpoint if present
TOK = 65536                # tokens per optimizer step
SEEDS = {R1: ["r1_s1_kda_hyb_seed43", "r1_s1_kda_hyb_seed44"],
         R2: ["r2_s1_gdn_hyb_seed43", "r2_s1_gdn_hyb_seed44"]}
MATH = [  # (model, baseline, what the paired difference isolates)
    ("r1_s1math_kda_hyb", R1, "stage-1 corpus: FineMath vs FineWeb (R1)"),
    ("r1_s2math_kda_hyb", "r1_s2_kda_hyb", "stage-2 corpus: FineMath vs FineWeb (R1-S2), same start R1"),
    ("r1_s2math_kda_hyb", R1, "FineMath stage 2 vs its start (R1)"),
    ("r1fm_s2math_kda_hyb", "r1_s2math_kda_hyb", "run B vs A: stage-1 corpus, given a FineMath stage 2"),
    ("r1fm_s2math_kda_hyb", "r1_s1math_kda_hyb", "run B vs its start (R1-S1-math)"),
    # v4: FineMath stage 2 on R2-R4 (each from its FineWeb stage 1, = R1-S2-math's design)
    ("r2_s2math_gdn_hyb", "r2_s2_gdn_hyb", "stage-2 corpus, GDN hybrid"),
    ("r3_s2math_kda_only", "r3_s2_kda_only", "stage-2 corpus, KDA all-linear"),
    ("r4_s2math_gdn_only", "r4_s2_gdn_only", "stage-2 corpus, GDN all-linear"),
    ("r1_s2math_kda_hyb", "r2_s2math_gdn_hyb", "MECHANISM within FineMath S2, hybrid: KDA - GDN (FineWeb S2: +17.9)"),
    ("r3_s2math_kda_only", "r4_s2math_gdn_only", "MECHANISM within FineMath S2, all-linear: KDA - GDN (FineWeb S2: +9.3)"),
    ("r3_s2math_kda_only", "r1_s2math_kda_hyb", "LAYOUT within FineMath S2, KDA: all-linear - hybrid (FineWeb S2: -1.9)"),
    ("r4_s2math_gdn_only", "r2_s2math_gdn_hyb", "LAYOUT within FineMath S2, GDN: all-linear - hybrid (FineWeb S2: +6.7)"),
]
MATH_MODELS = ["r1_s1math_kda_hyb", "r1_s2math_kda_hyb", "r1fm_s2math_kda_hyb",
               "r2_s2math_gdn_hyb", "r3_s2math_kda_only", "r4_s2math_gdn_only"]
PREREG_GAIN = 5.0          # points, R1-S2 over R1 (rev 36, carried into rev 42)


def main(d):
    data, _ = load(d)
    if TEACHER not in data:
        n = len(glob.glob(os.path.join(d, "*.jsonl")))
        sys.exit(f"no teacher results in {os.path.abspath(d)}  ({n} .jsonl files; models found: "
                 f"{sorted(data) or 'none'}).  Pass the arith_probe_v2 results directory.")
    print(f"results: {os.path.abspath(d)}   models: {len(data)}")
    conds = list(dict.fromkeys(k[0] for k in data[TEACHER]))
    keys = [k for k in data[TEACHER] if not k[0].startswith("copy")]

    def rows(m, c=None):
        return [r for k, r in data[m].items() if c is None or k[0] == c]

    def pooled(m):
        return 100 * acc([r for r in rows(m) if r["op"] != "copy"])[0]

    order = [m for m in [TEACHER, R1, R2, R3, R4, *S2, COS, *MATH_MODELS] if m in data]
    W = 10
    print("=" * 110)
    print("ACCURACY (%)")
    print(f"{'cond':10}" + "".join(f"{m[:W - 1]:>{W}}" for m in order))
    for c in conds:
        print(f"{c:10}" + "".join(f"{100 * acc(rows(m, c))[0]:{W}.1f}" for m in order))
    print(f"{'ARITH pool':10}" + "".join(f"{pooled(m):{W}.1f}" for m in order))

    print("\n" + "=" * 110)
    print("STAGE 2 - STAGE 1, same student, paired (points)        gap closure G = gain / (teacher - stage 1)")
    T = pooled(TEACHER)
    for s2, s1 in S2.items():
        if s2 in data and s1 in data:
            p, lo, hi = paired_boot(data[s1], data[s2], keys)
            G = (pooled(s2) - pooled(s1)) / (T - pooled(s1))
            line = f"    {s2:18} {pooled(s1):5.1f} -> {pooled(s2):5.1f}   {p:+6.1f} [{lo:+.1f}, {hi:+.1f}]   G = {G:+.2f}"
            if s1 == R1:
                ok = pooled(s2) >= pooled(R1) + PREREG_GAIN
                line += (f"   PRE-REG (>= R1 + {PREREG_GAIN:.0f} = {pooled(R1) + PREREG_GAIN:.1f}%): "
                         + ("✅ met" if ok else "❌ not met"))
            print(line)

    print("\nKDA - GDN within stage 2, paired (points)   [stage 1: +13.6 hybrid, +10.0 all-linear]")
    for kda, gdn, lay in (("r1_s2_kda_hyb", "r2_s2_gdn_hyb", "hybrid"), ("r3_s2_kda_only", "r4_s2_gdn_only", "all-linear")):
        if kda in data and gdn in data:
            p, lo, hi = paired_boot(data[gdn], data[kda], keys)
            print(f"    {lay:11} {p:+6.1f} [{lo:+.1f}, {hi:+.1f}]")

    print("\nLAYOUT within stage 2: all-linear - hybrid, paired (points)   [stage 1: ~0]")
    for a, h, mech in (("r3_s2_kda_only", "r1_s2_kda_hyb", "KDA"), ("r4_s2_gdn_only", "r2_s2_gdn_hyb", "GDN")):
        if a in data and h in data:
            p, lo, hi = paired_boot(data[h], data[a], keys)
            print(f"    {mech:4} {p:+6.1f} [{lo:+.1f}, {hi:+.1f}]")

    if COS in data and "r1_s2_kda_hyb" in data:
        p, lo, hi = paired_boot(data["r1_s2_kda_hyb"], data[COS], keys)
        print(f"\nCOSINE 5e-5 - CONSTANT 1e-5 (both R1, 1 B), paired: {p:+.1f} [{lo:+.1f}, {hi:+.1f}] points")

    rows_m = [(a, b, why) for a, b, why in MATH if a in data and b in data]
    if rows_m:
        print("\nFINEMATH, paired (points)            pooled a -> b is baseline -> model; G = gain / (teacher - baseline)")
        for a, b, why in rows_m:
            p, lo, hi = paired_boot(data[b], data[a], keys)
            G = (pooled(a) - pooled(b)) / (T - pooled(b))
            print(f"    {a:20} vs {b:16} {pooled(b):5.1f} -> {pooled(a):5.1f}   {p:+6.1f} [{lo:+.1f}, {hi:+.1f}]"
                  f"   G = {G:+.2f}   ({why})")

    def slope(pts):
        """least-squares points per 100 M tokens over [(step, model), ...]"""
        xs = [st_ * TOK / 1e8 for st_, _ in pts]
        ys = [pooled(m) for _, m in pts]
        if len(xs) < 3:
            return float("nan")
        mx, my = st.mean(xs), st.mean(ys)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)

    cc = [c for c in CURVE_CONDS if c in conds]
    by_step = {}
    for label, prefix, final in CURVES:
        ck = sorted(m for m in data if m.startswith(prefix))
        if not ck:
            continue
        pts = [(0, R1)] + [(int(m[len(prefix):]), m) for m in ck]
        if final in data:
            pts.append((15260, final))
        pts = [(s_, m) for s_, m in pts if m in data]
        by_step[label] = dict(pts)
        print(f"\nTOKEN CURVE  {label}   (pooled arithmetic; step x 65,536 tokens; step 0 = R1)")
        print(f"    {'step':>6} {'tokens':>7} {'pooled':>7}" + "".join(f"{c:>8}" for c in cc) + "   model")
        for s_, m in pts:
            print(f"    {s_:6} {s_ * TOK / 1e6:6.0f}M {pooled(m):7.1f}"
                  + "".join(f"{100 * acc(rows(m, c))[0]:8.1f}" for c in cc) + f"   {m}")
        late = [(s_, m) for s_, m in pts if s_ >= 4578]
        first = next((m for s_, m in pts if s_ == 1526), None)
        print(f"    slope: {slope(pts):+.2f} points / 100 M over 0-1 B,  {slope(late):+.2f} over 300 M-1 B"
              + (f";  first 100 M: {pooled(first) - pooled(R1):+.1f}" if first else ""))
    if len(by_step) == 2:
        (la, A), (lb, B) = by_step.items()
        common = sorted(set(A) & set(B) - {0})
        if common:
            print(f"\n    FineMath - FineWeb at equal tokens, paired (points)")
            for s_ in common:
                p, lo, hi = paired_boot(data[A[s_]], data[B[s_]], keys)
                print(f"    step {s_:6} {s_ * TOK / 1e6:6.0f}M   {pooled(A[s_]):5.1f} -> {pooled(B[s_]):5.1f}"
                      f"   {p:+6.1f} [{lo:+.1f}, {hi:+.1f}]")

    print("\nSEED SPREAD (stage 1, data order only)")
    for base, reps in SEEDS.items():
        ms = [m for m in [base, *reps] if m in data]
        if len(ms) > 1:
            v = [pooled(m) for m in ms]
            print(f"    {base:16} pooled {', '.join(f'{x:.1f}' for x in v)}   SD {st.stdev(v):.2f}")
    pairs = [(R1, R2)] + list(zip(SEEDS[R1], SEEDS[R2]))
    gaps = [pooled(a) - pooled(b) for a, b in pairs if a in data and b in data]
    if len(gaps) > 1:
        print(f"    KDA - GDN (hybrid) per seed: {', '.join(f'{g:+.1f}' for g in gaps)}   "
              f"mean {st.mean(gaps):+.1f} +- {st.stdev(gaps):.1f}")


if __name__ == "__main__":
    default = os.path.join(os.environ.get("WORK_LUKAS", "/leonardo_work/EUHPC_D34_049/WORK_LUKAS"),
                           "eval_results", "arith_probe_v2")
    main(sys.argv[1] if len(sys.argv) > 1 else default)