#!/usr/bin/env python
"""Per-sequence view of heldout_kl.py's results -- no GPU, run it on the login node.

    python heldout_kl_seqs.py r1_s1math_kda_hyb r1_kda_hybrid r1_s2math_kda_hyb [r5_kda_hybrid_1ep]
    python heldout_kl_seqs.py ... --pairs r1_s2_kda_hyb:r3_s2_kda_only r1_kda_hybrid:r3_kda_only
    python heldout_kl_seqs.py r1_s1math_kda_hyb --text 4      # also print the start of its 4 worst sequences
    LAYOUT=docs python heldout_kl_seqs.py ...                # the one-document-per-sequence results (v2)

Reads $WORK_LUKAS/eval_results/heldout_kl/<name>.json (per_seq_ce, per_seq_kl; the teacher's
per_seq_ce gives the per-sequence gap).  The 64 sequences are identical for every model, so
sequence i can be compared across models.

  1. robust summaries per model and corpus: mean / MEDIAN / trimmed KL (without the 5 worst
     sequences by CE gap) -- if the median is near the training KL and the mean is not, the mean
     is carried by a few sequences (R5's signature), not by uniform damage
  2. the bad sequences (CE gap to the teacher > 0.5 nat) of the FIRST model, with the gap every
     other listed model has on the SAME sequence -> damage specific to one model, or hard text?
  3. --pairs A:B -> paired per-sequence KL difference B - A, mean +- SE, and how many of 64 favour B
  4. --text N -> the first 300 characters of the first model's N worst sequences per corpus
     (rebuilds the sequences exactly as heldout_kl.py does; needs only the tokenizer)
"""
import argparse
import json
import math
import os
import statistics as st

W = os.environ.get("WORK_LUKAS", "/leonardo_work/EUHPC_D34_049/WORK_LUKAS")
_LAY = os.environ.get("LAYOUT", "concat")   # docs / concat_bos / concat_eos: heldout_kl.py v2-v3
OUTDIR = f"{W}/eval_results/heldout_kl" + ("" if _LAY == "concat" else f"_{_LAY}")
CORP = ("web", "math")


def get(name):
    p = f"{OUTDIR}/{name}.json"
    if not os.path.exists(p):
        raise SystemExit(f"no {p} -- run heldout_kl.sbatch for it first (MODELS_FILE for extra models)")
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--text", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--weighted", action="store_true",
                    help="LAYOUT=docs only: token-weighted means and short vs long documents")
    a = ap.parse_args()
    T = get("teacher")
    R = {m: get(m) for m in a.models}
    for m in [x for p in a.pairs for x in p.split(":")]:
        R.setdefault(m, get(m))

    def gaps(m, c):
        return [s - t for s, t in zip(R[m][c]["per_seq_ce"], T[c]["per_seq_ce"])]

    print("1. KL (T = 2) per sequence: mean / median / trimmed (5 worst by CE gap removed);  CE gap mean / median")
    print(f"   {'model':24}" + "".join(f"{c + ' KL mean':>14}{'median':>8}{'trim':>8}{'gap':>8}{'gap med':>9}" for c in CORP))
    for m in a.models:
        line = f"   {m[:23]:24}"
        for c in CORP:
            kl, g = R[m][c]["per_seq_kl"], gaps(m, c)
            keep = sorted(range(len(g)), key=lambda i: g[i])[:-5]
            line += (f"{st.mean(kl):14.4f}{st.median(kl):8.4f}{st.mean(kl[i] for i in keep):8.4f}"
                     f"{st.mean(g):+8.3f}{st.median(g):+9.3f}")
        print(line)

    first = a.models[0]
    for c in CORP:
        g0 = gaps(first, c)
        bad = sorted((i for i in range(len(g0)) if g0[i] > a.thr), key=lambda i: -g0[i])
        print(f"\n2. {c}: {len(bad)} sequences with CE gap > {a.thr} nat in {first}; gap of each listed model on them")
        if bad:
            print(f"   {'seq':>5}" + "".join(f"{m[:15]:>16}" for m in a.models))
            G = {m: gaps(m, c) for m in a.models}
            for i in bad:
                print(f"   {i:5}" + "".join(f"{G[m][i]:+16.3f}" for m in a.models))
            for m in a.models[1:]:
                n = sum(G[m][i] > a.thr for i in bad)
                print(f"   {m}: also > {a.thr} on {n} / {len(bad)} of them; worst overall {max(G[m]):+.3f}")

    if a.pairs:
        print("\n3. paired per-sequence KL difference B - A (negative = B closer to the teacher)")
        for p in a.pairs:
            A, B = p.split(":")
            for c in CORP:
                d = [b - x for x, b in zip(R[A][c]["per_seq_kl"], R[B][c]["per_seq_kl"])]
                se = st.stdev(d) / math.sqrt(len(d))
                print(f"   {B:20} - {A:20} {c:4}  {st.mean(d):+.4f} +- {se:.4f}  (z {st.mean(d) / se:+.1f}),"
                      f"  B better on {sum(x < 0 for x in d)} / {len(d)},  median {st.median(d):+.4f}")

    if a.weighted:
        # docs layout: documents differ in length, and heldout_kl averages PER DOCUMENT, so a short
        # document counts as much as a 2048-token one.  DistillKit's eval_loss is per packed 2048-token
        # row = effectively token-weighted.  If a model is bad mainly early in a sequence (short
        # context), the per-document mean exaggerates it relative to DistillKit's number.
        assert _LAY == "docs", "--weighted needs LAYOUT=docs"
        from transformers import AutoTokenizer
        from heldout_kl import CORPORA, TEACHER, build
        tok = AutoTokenizer.from_pretrained(TEACHER)
        print("\n5. docs layout: per-document mean vs TOKEN-weighted mean of KL; short (< 512 tokens) vs long documents")
        for c, path in CORPORA.items():
            n = [len(x) - 1 for x in build(tok, path)]          # predicted positions per document
            short = [i for i in range(len(n)) if n[i] < 512]
            long_ = [i for i in range(len(n)) if n[i] >= 512]
            print(f"   {c}: {len(short)} short / {len(long_)} long documents, {sum(n):,} predicted tokens")
            print(f"   {'model':24}{'per-doc':>9}{'token-wt':>10}{'short':>8}{'long':>8}")
            for m in a.models:
                k = R[m][c]["per_seq_kl"]
                tw = sum(x * w for x, w in zip(k, n)) / sum(n)
                ms = st.mean(k[i] for i in short) if short else float("nan")
                ml = st.mean(k[i] for i in long_) if long_ else float("nan")
                print(f"   {m[:23]:24}{st.mean(k):9.4f}{tw:10.4f}{ms:8.4f}{ml:8.4f}")

    if a.text:
        from transformers import AutoTokenizer
        from heldout_kl import CORPORA, TEACHER, build
        tok = AutoTokenizer.from_pretrained(TEACHER)
        for c, path in CORPORA.items():
            b = build(tok, path)
            g0 = gaps(first, c)
            print(f"\n4. {c}: the {a.text} worst sequences of {first}")
            for i in sorted(range(len(g0)), key=lambda i: -g0[i])[:a.text]:
                txt = tok.decode(b[i][:120]).replace("\n", " | ")
                print(f"   seq {i} (gap {g0[i]:+.2f}): {txt[:300]}")


if __name__ == "__main__":
    main()