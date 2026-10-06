#!/usr/bin/env python
"""How much did batching move the ANSWERS?  Compare two arith_probe runs.

    python arith_agree.py BATCHED_DIR BS1_DIR r1_kda_hybrid r5_kda_hybrid_1ep
"""
import json
import sys

a_dir, b_dir, models = sys.argv[1], sys.argv[2], sys.argv[3:]


def load(d, m):
    return {(r["cond"], r["idx"]): r for r in map(json.loads, open(f"{d}/{m}.jsonl"))}


pool = {}
for m in models:
    A, B = load(a_dir, m), load(b_dir, m)
    keys = [k for k in A if k in B]
    same = sum(A[k]["pred"] == B[k]["pred"] for k in keys)
    ar = [k for k in keys if A[k]["op"] != "copy"]
    acc_a = 100 * sum(A[k]["correct"] for k in ar) / len(ar)
    acc_b = 100 * sum(B[k]["correct"] for k in ar) / len(ar)
    flips = [(k, A[k]["pred"], B[k]["pred"], A[k]["target"]) for k in keys if A[k]["pred"] != B[k]["pred"]]
    pool[m] = (acc_a, acc_b)
    print(f"{m:20} identical answers {same}/{len(keys)} ({100 * same / len(keys):.2f}%)   "
          f"pooled arithmetic batched {acc_a:.1f}%  vs  batch-1 {acc_b:.1f}%  ({acc_b - acc_a:+.1f})")
    for k, pa, pb, t in flips[:8]:
        print(f"    {k[0]:8} #{k[1]:<4} batched {pa}  batch-1 {pb}  target {t}")
if len(models) == 2:
    (a1, b1), (a5, b5) = pool[models[0]], pool[models[1]]
    print(f"\n{models[1]} - {models[0]}:  batched {a5 - a1:+.1f}   batch-1 {b5 - b1:+.1f}")