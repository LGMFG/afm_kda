#!/usr/bin/env python
"""Read arith_probe.py's results and score the PRE-REGISTERED predictions.

    python arith_compare.py $WORK_LUKAS/arith_probe_v2

Every verdict below was written down in claude/first_distillkit_runs.md
(revision 34) BEFORE any of these models was run through arith_probe.py.
The thresholds live in this file as constants so that checking is matching,
not judging.  Do not edit them after seeing results; if one turns out to be
badly chosen, say so in the doc and keep the original verdict alongside.

Paired statistics throughout: every model saw byte-identical prompts, so a
difference is computed per problem and bootstrapped WITHIN each condition.
"""
import glob
import json
import os
import sys

# ---- names as written by arith_probe.sbatch ------------------------------
TEACHER = "teacher"
R1, R2, R3, R4, R5 = ("r1_kda_hybrid", "r2_gdn_hybrid", "r3_kda_only",
                      "r4_gdn_only", "r5_kda_hybrid_1ep")
CURVE_PREFIX = "r5_ck"
R5_FINAL_CKPT = "r5_ck34940"

# ---- pre-registered thresholds (revision 34) -----------------------------
COPY_OK = 0.95          # P1: reading intact if every student >= this on copy_1..3 pooled
TEACHER_COPY_MIN = 0.95 # P0: probe valid only if the teacher copies
SWEEP_DROP = 0.15       # P2: student/teacher ratio add_4 at least this far below add_2
CARRY_EXCESS = 0.10     # P3: student's carry penalty exceeds the teacher's by this
R5_BAND = 5.0           # P5: points; see verdict text
R5_CLOSURE = 0.25       # P5: gap closure needed to call it "recovery"
CURVE_FLAT = 5.0        # P6: points; max-min across R5's checkpoints
N_BOOT = 10_000
SEED = 1


def load(d):
    data = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        if ".smoke." in f:
            continue
        rows = [json.loads(x) for x in open(f)]
        data[rows[0]["model"]] = {(r["cond"], r["idx"]): r for r in rows}
    summ = {}
    for f in glob.glob(os.path.join(d, "*.summary.json")):
        if ".smoke." not in f:
            s = json.load(open(f))
            summ[s["model"]] = s
    return data, summ


def acc(rows, pred=lambda r: True):
    sel = [r for r in rows if pred(r)]
    return (sum(r["correct"] for r in sel) / len(sel), len(sel)) if sel else (float("nan"), 0)


def paired_boot(A, B, keys, n=N_BOOT, seed=SEED):
    """mean(B - A) in points, with a 95% CI stratified by condition."""
    import numpy as np
    rng = np.random.default_rng(seed)
    by_cond = {}
    for k in keys:
        by_cond.setdefault(k[0], []).append(int(B[k]["correct"]) - int(A[k]["correct"]))
    tot = sum(len(v) for v in by_cond.values())
    point = 100 * sum(sum(v) for v in by_cond.values()) / tot
    sums = np.zeros(n)
    for v in by_cond.values():
        v = np.asarray(v)
        sums += v[rng.integers(0, len(v), size=(n, len(v)))].sum(1)
    stats = np.sort(100 * sums / tot)
    return point, stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def main(d):
    data, summ = load(d)
    if TEACHER not in data:
        sys.exit("no teacher results -- nothing is interpretable without the teacher")
    order = [m for m in (TEACHER, R1, R2, R3, R4, R5) if m in data]
    curve = sorted(m for m in data if m.startswith(CURVE_PREFIX))
    conds = []
    for k in data[TEACHER]:
        if k[0] not in conds:
            conds.append(k[0])
    arith = [c for c in conds if not c.startswith("copy")]
    copies = [c for c in conds if c.startswith("copy")]

    def rows(m, c=None):
        return [r for k, r in data[m].items() if c is None or k[0] == c]

    W = 9
    print("=" * 100)
    print("ACCURACY (%) -- exact match on the first integer")
    print(f"{'cond':10}{'n':>5}" + "".join(f"{m[:W-1]:>{W}}" for m in order))
    for c in conds:
        n = len(rows(TEACHER, c))
        print(f"{c:10}{n:5}" + "".join(f"{100 * acc(rows(m, c))[0]:{W}.1f}" for m in order))
    pooled = {m: acc([r for r in rows(m) if r["op"] != "copy"])[0] for m in order}
    print(f"{'ARITH pool':10}{len(arith_rows := [r for r in rows(TEACHER) if r['op'] != 'copy']):5}"
          + "".join(f"{100 * pooled[m]:{W}.1f}" for m in order))

    # ---- validity ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("P0  VALIDITY")
    # pooled over copy_1..3 (n = 5 + 86 + 200): copy_1 alone has n = 5, where
    # one miss is 80 % -- too fragile to gate anything on by itself
    c13 = lambda m: acc([r for r in rows(m) if r["op"] == "copy" and r["d"] in ("1", "2", "3")])[0]
    tcopy = c13(TEACHER)
    ok0 = tcopy >= TEACHER_COPY_MIN
    print(f"    teacher, copy_1..3 pooled = {100 * tcopy:.1f}%   (need >= {100 * TEACHER_COPY_MIN:.0f})"
          f"   {'✅' if ok0 else '❌ the probe format itself is broken -- STOP'}")
    for m in order + curve:
        mm = summ.get(m, {}).get("batch_vs_single_mismatches")
        if mm:
            print(f"    ⚠️⚠️ {m}: {mm} batch-vs-single mismatches -- its numbers are suspect")
    tk = summ.get(TEACHER, {}).get("tokenisation", {})
    if tk:
        print("    tokens per number: " + "  ".join(
            f"{k}d={v['mean_tokens']:.2f}" for k, v in sorted(tk.items())))
        if tk.get("3", {}).get("mean_tokens", 3) < 1.5:
            print("    ⚠️⚠️ 3-digit numbers are ~ONE token.  For d <= 3 there is no per-digit")
            print("       alignment in the prompt; read the sweep as a VOCABULARY effect there.")

    students = [m for m in order if m != TEACHER]

    # ---- P1 reading -------------------------------------------------------
    print("\nP1  READING INTACT?   every student >= 95% on copy_1..3 pooled (n = 291)")
    for m in students:
        lo = c13(m)
        c4 = acc(rows(m, "copy_4"))[0]
        print(f"    {m:20} copy_1..3 = {100 * lo:5.1f}%   copy_4 = {100 * c4:5.1f}%   "
              + ("✅ reads fine -> the failure is COMPUTATION" if lo >= COPY_OK
                 else "❌ READING IS BROKEN -> 'arithmetic circuit destroyed' must be revised"))

    # ---- P2 digit sweep ---------------------------------------------------
    print("\nP2  DIGIT SWEEP   student/teacher on add_d; predicted to FALL with d")
    t = {dd: acc(rows(TEACHER, f"add_{dd}"))[0] for dd in (1, 2, 3, 4)}
    print(f"    {'':20}" + "".join(f"{'add_' + str(dd):>9}" for dd in (1, 2, 3, 4)))
    print(f"    {'teacher (abs %)':20}" + "".join(f"{100 * t[dd]:9.1f}" for dd in (1, 2, 3, 4)))
    for m in students:
        rat = {dd: acc(rows(m, f"add_{dd}"))[0] / t[dd] if t[dd] else float("nan")
               for dd in (1, 2, 3, 4)}
        fell = rat[4] <= rat[2] - SWEEP_DROP
        print(f"    {m + ' (ratio)':20}" + "".join(f"{rat[dd]:9.2f}" for dd in (1, 2, 3, 4))
              + f"   {'✅ falls' if fell else '❌ does not fall by ' + str(SWEEP_DROP)}")

    # ---- P3 carry ---------------------------------------------------------
    print("\nP3  CARRY / BORROW   penalty = no-carry% minus carry%, add_2..4 pooled")
    def pen(m, op, yes, no):
        R = [r for r in rows(m) if r["op"] == op and r["d"] in ("2", "3", "4")]
        return acc(R, lambda r: r["cls"] == no)[0] - acc(R, lambda r: r["cls"] == yes)[0]
    tp_a, tp_s = pen(TEACHER, "add", "carry", "nocarry"), pen(TEACHER, "sub", "borrow", "noborrow")
    print(f"    {'teacher':20} carry penalty {100 * tp_a:+6.1f}   borrow penalty {100 * tp_s:+6.1f}")
    for m in students:
        pa, ps = pen(m, "add", "carry", "nocarry"), pen(m, "sub", "borrow", "noborrow")
        v = pa - tp_a >= CARRY_EXCESS
        print(f"    {m:20} carry penalty {100 * pa:+6.1f}   borrow penalty {100 * ps:+6.1f}   "
              f"{'✅ carry hurts the student more' if v else '❌ no excess carry penalty'}")

    # ---- error anatomy ----------------------------------------------------
    print("\n    ERROR ANATOMY of wrong arithmetic answers (descriptive, not pre-registered)")
    print(f"    {'':20}{'wrong':>7}{'no number':>11}{'= operand':>11}{'units ok':>10}")
    for m in order:
        W_ = [r for r in rows(m) if r["op"] != "copy" and not r["correct"]]
        if not W_:
            continue
        nonum = sum(r["pred"] is None for r in W_)
        opc = sum(r["pred"] is not None and r["pred"] in (r["a"], r["b"]) for r in W_)
        units = sum(r["pred"] is not None and abs(r["pred"]) % 10 == abs(r["target"]) % 10 for r in W_)
        n = len(W_)
        print(f"    {m:20}{n:7}{100 * nonum / n:10.1f}%{100 * opc / n:10.1f}%{100 * units / n:9.1f}%")
    print("    'units ok' = right last digit, wrong number: the local column worked, the")
    print("    positional/carry part did not.  Random guessing gives ~10%.")

    keys = [k for k in data[TEACHER] if k[0] in arith]

    # ---- P4 mechanism -----------------------------------------------------
    print("\nP4  MECHANISM   KDA - GDN on the pooled arithmetic, paired 95% CI (points)")
    for kda, gdn, lay in ((R1, R2, "hybrid"), (R3, R4, "all-linear")):
        if kda in data and gdn in data:
            p, lo, hi = paired_boot(data[gdn], data[kda], keys)
            print(f"    {lay:11} {p:+6.1f}  [{lo:+.1f}, {hi:+.1f}]   "
                  + ("✅ KDA ahead" if lo > 0 else "❌ not separated" if hi > 0 else "❌ GDN AHEAD"))

    # ---- P5 the R5 question ----------------------------------------------
    if R1 in data and R5 in data:
        print("\nP5  ⭐ DID 7.6x MORE FINEWEB-EDU TOKENS RECOVER ARITHMETIC?   R5 - R1, pooled")
        p, lo, hi = paired_boot(data[R1], data[R5], keys)
        T_, A_, B_ = pooled[TEACHER], pooled[R1], pooled[R5]
        G = (B_ - A_) / (T_ - A_) if T_ != A_ else float("nan")
        print(f"    teacher {100 * T_:.1f}%   R1 {100 * A_:.1f}%   R5 {100 * B_:.1f}%")
        print(f"    R5 - R1 = {p:+.1f} points, 95% CI [{lo:+.1f}, {hi:+.1f}];  gap closure G = {G:+.2f}")
        if lo > 0 and G >= R5_CLOSURE:
            v = "RECOVERY -- the token budget WAS binding for arithmetic"
        elif lo > 0:
            v = "SMALL GAIN -- real but under a quarter of the gap; the corpus is still binding"
        elif hi < 0:
            v = "EROSION -- longer training on FineWeb-Edu made arithmetic WORSE"
        elif lo >= -R5_BAND and hi <= R5_BAND:
            # an EQUIVALENCE statement, not just "CI includes zero": the whole
            # interval sits inside +-R5_BAND points
            v = (f"NULL -- any change is within +-{R5_BAND:.0f} points; the CORPUS is "
                 "binding, not the budget")
        else:
            v = "INCONCLUSIVE -- CI includes 0 but reaches past +-5 points; cannot call"
        print(f"    VERDICT: {v}")
        print("    Pre-registered prediction (rev 34): NULL, with EROSION the next most likely.")

    # ---- P6 the curve -----------------------------------------------------
    if curve:
        print("\nP6  R5 CHECKPOINT CURVE   pooled arithmetic vs tokens")
        ys = []
        for m in curve:
            step = int(m[len(CURVE_PREFIX):])
            a = pooled.get(m, acc([r for r in rows(m) if r["op"] != "copy"])[0])
            ys.append(100 * a)
            print(f"    step {step:6}  {step * 65536 / 1e9:5.2f} B tokens   {100 * a:5.1f}%")
        rng_ = max(ys) - min(ys)
        print(f"    range {rng_:.1f} points   "
              + ("✅ FLAT, as predicted" if rng_ < CURVE_FLAT else "❌ NOT flat -- look at the shape"))
        if R5_FINAL_CKPT in data and R5 in data:
            same = sum(data[R5][k]["pred"] == data[R5_FINAL_CKPT][k]["pred"] for k in data[R5])
            print(f"    identity check  top-level vs {R5_FINAL_CKPT}: {same}/{len(data[R5])} identical predictions"
                  + ("  ✅" if same == len(data[R5]) else "  ⚠️ the two 'identical' weights disagree"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")