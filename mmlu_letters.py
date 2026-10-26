#!/usr/bin/env python
"""Is the R1 -> R1-S2 MMLU gain knowledge, or a change in answer-letter bias?

    python mmlu_letters.py <samples dir A> <samples dir B> [--names R1 R1-S2]

Each dir = the lm-eval output folder with samples_mmlu_*.jsonl (written by --log_samples).
Near chance (0.25), accuracy can rise just because a model shifts which letter it prefers
(if the gold answers are not uniformly spread, or if a strong bias is removed). Five checks:

  1. letter distribution of predictions vs gold     -> a bias shows up as one letter dominating
  2. accuracy per gold letter                       -> a bias-driven gain sits on one letter
  3. letter-CALIBRATED accuracy: subtract each model's mean log-likelihood per letter
     (its letter prior) before the argmax           -> a knowledge gain survives, a bias gain shrinks
  4. mean probability on the gold answer (softmax over A-D) and paired McNemar on items
  5. per-subject change: how many of the 57 subjects improve (sign test)

Reading:  gain survives calibration (3) + p(gold) rises (4) + most subjects improve (5)
          -> the model knows more.   Gain vanishes in (3) and sits on one letter (1, 2)
          -> letter-bias artefact.
"""
import argparse
import glob
import json
import math
import os
import re
import statistics as st
from collections import Counter

L = "ABCD"


def lls(rec):
    fr = rec.get("filtered_resps")
    if fr is not None:
        return [float(x[0] if isinstance(x, (list, tuple)) else x) for x in fr]
    return [float(r[0][0]) for r in rec["resps"]]


def load(d):
    files = sorted(glob.glob(os.path.join(d, "**", "samples_mmlu_*.jsonl"), recursive=True))
    if not files:
        raise SystemExit(f"no samples_mmlu_*.jsonl under {d}")
    data = {}
    for f in files:
        subj = re.sub(r"^samples_mmlu_|_\d{4}-\d\d-\d\dT.*$", "", os.path.basename(f))
        for line in open(f):
            r = json.loads(line)
            ll = lls(r)
            if len(ll) != 4:
                continue
            data[(subj, int(r["doc_id"]))] = (ll, int(r["target"]))
    return data


def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def summarise(name, data):
    keys = sorted(data)
    pred = {k: max(range(4), key=lambda i: data[k][0][i]) for k in keys}
    acc = st.mean(pred[k] == data[k][1] for k in keys)
    # letter prior = mean log-likelihood per letter over all items; calibrated argmax
    prior = [st.mean(data[k][0][i] for k in keys) for i in range(4)]
    cal = {k: max(range(4), key=lambda i: data[k][0][i] - prior[i]) for k in keys}
    acc_cal = st.mean(cal[k] == data[k][1] for k in keys)
    pgold = st.mean(softmax(data[k][0])[data[k][1]] for k in keys)
    cnt = Counter(pred.values())
    per_gold = {}
    for i in range(4):
        ks = [k for k in keys if data[k][1] == i]
        per_gold[i] = st.mean(pred[k] == i for k in ks) if ks else float("nan")
    return dict(name=name, keys=keys, pred=pred, cal=cal, acc=acc, acc_cal=acc_cal, pgold=pgold,
                cnt=cnt, per_gold=per_gold, n=len(keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    a = ap.parse_args()
    da, db = load(a.a), load(a.b)
    common = sorted(set(da) & set(db))
    print(f"items: {len(da)} / {len(db)}, common {len(common)}")
    da = {k: da[k] for k in common}
    db = {k: db[k] for k in common}
    assert all(da[k][1] == db[k][1] for k in common), "gold answers differ -- not the same items"
    A, B = summarise(a.names[0], da), summarise(a.names[1], db)
    gold = Counter(da[k][1] for k in common)

    print("\n1. LETTER DISTRIBUTION (% of items)            A      B      C      D")
    print(f"   {'gold':40}" + "".join(f"{100 * gold[i] / len(common):7.1f}" for i in range(4)))
    for M in (A, B):
        print(f"   {M['name'] + ' predictions':40}" + "".join(f"{100 * M['cnt'][i] / M['n']:7.1f}" for i in range(4)))

    print("\n2. ACCURACY PER GOLD LETTER (%)                 A      B      C      D")
    for M in (A, B):
        print(f"   {M['name']:40}" + "".join(f"{100 * M['per_gold'][i]:7.1f}" for i in range(4)))

    print("\n3./4. OVERALL                       acc     letter-calibrated acc    mean p(gold)")
    for M in (A, B):
        print(f"   {M['name']:30}{100 * M['acc']:7.2f}{100 * M['acc_cal']:18.2f}{M['pgold']:20.4f}")
    print(f"   {'difference B - A':30}{100 * (B['acc'] - A['acc']):+7.2f}{100 * (B['acc_cal'] - A['acc_cal']):+18.2f}"
          f"{B['pgold'] - A['pgold']:+20.4f}")
    b_only = sum(B["pred"][k] == db[k][1] and A["pred"][k] != da[k][1] for k in common)
    a_only = sum(A["pred"][k] == da[k][1] and B["pred"][k] != db[k][1] for k in common)
    z = (b_only - a_only) / math.sqrt(b_only + a_only) if b_only + a_only else 0
    same = sum(A["pred"][k] == B["pred"][k] for k in common)
    print(f"   paired: right only in {B['name']} {b_only}, only in {A['name']} {a_only}  -> McNemar z = {z:+.1f};"
          f"  same prediction on {100 * same / len(common):.1f} % of items")

    print("\n5. PER SUBJECT (accuracy change B - A)")
    subj = sorted({k[0] for k in common})
    deltas = []
    for s in subj:
        ks = [k for k in common if k[0] == s]
        deltas.append((st.mean(B["pred"][k] == db[k][1] for k in ks) - st.mean(A["pred"][k] == da[k][1] for k in ks), s, len(ks)))
    up = sum(d > 0 for d, _, _ in deltas)
    down = sum(d < 0 for d, _, _ in deltas)
    zs = (up - down) / math.sqrt(up + down) if up + down else 0
    print(f"   {up} of {len(subj)} subjects up, {down} down (sign test z = {zs:+.1f})")
    deltas.sort()
    print("   largest drops: " + ", ".join(f"{s} {100 * d:+.0f} (n={n})" for d, s, n in deltas[:3]))
    print("   largest gains: " + ", ".join(f"{s} {100 * d:+.0f} (n={n})" for d, s, n in deltas[-3:][::-1]))


if __name__ == "__main__":
    main()