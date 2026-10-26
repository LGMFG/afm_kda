#!/usr/bin/env python
"""FineMath-4plus for DistillKit: download a pinned subset, then write THREE splits.

    <out>_heldout   2,000 docs, carved first, never trained on (exact-text leaks removed)
    <out>_stage1    >= --stage1-tokens effective tokens (default 300 M)  -> a stage-1 run
    <out>_stage2    >= --stage2-tokens effective tokens (default 1.0 B)  -> a stage-2 run
    <out>_stats.json

The three are pairwise disjoint by exact text (sha1), and stage1 / stage2 contain no
repeated text internally -- the same guarantees make_stage2_split.py v4 gave FineWeb-Edu.
Full instructions: claude/download_finemath.md.

Same recipe as FineWeb-Edu (download_fineweb_tokens_lowmem.py + prepare_fineweb_dataset.py):

  download   pin the dataset revision (REVISION file, reused on resubmission); seeded
             random choice of shards (shuffle the shard list, take shards in that order);
             calibrate effective AFM tokens/doc on the first chosen shard (random rows from
             random row groups -- low memory, unlike FineWeb's head-of-shard probe that ran
             9 % long); keep downloading until the docs cover both splits x --margin
             (row counts from parquet footers); write data_manifest.json.
  prepare    load, keep `text`; exact Llama-token count (`token_count` column); effective
             AFM tokens/doc K on 5,000 random docs; held-out split; one seeded permutation
             of the rest -> stage1 = first n1 distinct texts, stage2 = next n2 texts not in
             stage1; n = ceil(tokens x margin / K); everything is checked BEFORE anything
             is written; splits written in source order (sequential writes).

"Effective tokens" = what DistillKit trains on: TRL appends EOS and truncates every document
at 2048 before packing, so K = mean(min(afm_tokens + 1, 2048)). FineWeb-Edu check: this gives
~788 from the 2,000-doc truncation sample against 784.1 measured by the R5 warm-up.

FineMath (dataset card, HuggingFaceTB/finemath): finemath-4plus = 6.7 M docs, 9.6 B Llama
tokens; decontaminated against GSM8K, MATH, MMLU and ARC test sets (13-gram); ODC-By 1.0.

Usage (lrd_all_serial; see get_finemath.sbatch):
  python get_finemath.py download --out $WORK_LUKAS/datasets/finemath_4plus_raw
  python get_finemath.py prepare  --raw $WORK_LUKAS/datasets/finemath_4plus_raw \\
                                  --out $WORK_LUKAS/datasets/finemath_4plus --num-proc 4
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random

CONFIG_PREFIX = "finemath-4plus/"
REPO = "HuggingFaceTB/finemath"
TOKENIZER = "arcee-ai/AFM-4.5B-Base"


def effective(afm_lens, seq):
    """tokens DistillKit trains on per doc: EOS appended, truncated at seq."""
    return sum(min(n + 1, seq) for n in afm_lens) / len(afm_lens)


# --------------------------------------------------------------------------- download
def download(a):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    os.makedirs(a.out, exist_ok=True)
    api = HfApi()
    rev_file = os.path.join(a.out, "REVISION")
    if a.revision:
        rev = a.revision
    elif os.path.exists(rev_file):
        rev = open(rev_file).read().strip()
        print(f"re-using pinned revision from {rev_file}")
    else:
        rev = api.repo_info(REPO, repo_type="dataset").sha
    with open(rev_file, "w") as f:
        f.write(rev + "\n")
    files = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset", revision=rev)
                   if f.startswith(CONFIG_PREFIX) and f.endswith(".parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {CONFIG_PREFIX} at revision {rev}")
    print(f"revision {rev[:12]}: {len(files)} shards under {CONFIG_PREFIX}")
    order = files[:]
    random.Random(a.seed).shuffle(order)       # = random.sample(files, k) for any k

    def get(f):
        p = hf_hub_download(REPO, f, repo_type="dataset", revision=rev, local_dir=a.out)
        return p, pq.ParquetFile(p).metadata.num_rows          # footer only

    # calibrate on the first chosen shard: random rows from random row groups
    p0, n0 = get(order[0])
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pf = pq.ParquetFile(p0)
    rng = random.Random(a.seed)
    groups = rng.sample(range(pf.num_row_groups), min(20, pf.num_row_groups))
    per = math.ceil(a.probe_docs / len(groups))
    sample, llama = [], []
    for g in groups:
        t = pf.read_row_group(g, columns=["text", "token_count"])
        idx = rng.sample(range(t.num_rows), min(per, t.num_rows))
        sample += [t.column("text")[i].as_py() for i in idx]
        llama += [t.column("token_count")[i].as_py() for i in idx]
        del t
    afm = [len(x) for x in tok(sample, add_special_tokens=False)["input_ids"]]
    K = effective(afm, a.seq_len)
    over = sum(1 for x in afm if x + 1 > a.seq_len) / len(afm)
    need_tok = (a.stage1_tokens + a.stage2_tokens) * a.margin
    target = a.target_docs or math.ceil(need_tok / K) + a.heldout_docs + a.heldout_spare
    print(f"calibration, {len(afm)} random docs of {os.path.basename(p0)}: AFM {sum(afm) / len(afm):.0f} "
          f"tok/doc (Llama {sum(llama) / len(llama):.0f}); {over:.1%} docs truncated at {a.seq_len}; "
          f"effective K = {K:.0f} tok/doc")
    print(f"target: ({a.stage1_tokens:.2e} + {a.stage2_tokens:.2e}) x {a.margin} / K + held-out "
          f"= {target:,} docs" + ("   (--target-docs override)" if a.target_docs else ""))

    chosen, rows, nbytes = [], 0, 0
    for i, f in enumerate(order):
        p, n = (p0, n0) if i == 0 else get(f)
        chosen.append({"file": f, "rows": n})
        rows += n
        nbytes += os.path.getsize(p)
        print(f"  {len(chosen):3}  {f}  {n:,} rows   total {rows:,} docs, {nbytes / 2**30:.2f} GiB", flush=True)
        if rows >= target:
            break
    if rows < target:
        print(f"!! only {rows:,} docs in the whole config")

    man = {"repo": REPO, "revision": rev, "config_prefix": CONFIG_PREFIX,
           "shard_order_seed": a.seed, "n_shards_in_config": len(files),
           "target_docs": target, "n_shards": len(chosen), "n_docs": rows, "bytes": nbytes,
           "shards": chosen, "tokenizer": a.tokenizer, "seq_len": a.seq_len,
           "probe_docs": len(afm), "afm_tokens_per_doc_probe": sum(afm) / len(afm),
           "effective_tokens_per_doc_probe": K, "frac_docs_truncated": over,
           "stage1_tokens": a.stage1_tokens, "stage2_tokens": a.stage2_tokens, "margin": a.margin}
    with open(os.path.join(a.out, "data_manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(f"manifest -> {a.out}/data_manifest.json")


# --------------------------------------------------------------------------- prepare
def _hash(batch):
    return {"h": [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in batch["text"]]}


def prepare(a):
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer

    sufs = ("_heldout", "_stage1", "_stage2")
    for d in [a.out + s for s in sufs] + [a.out + "_stats.json"]:
        if os.path.exists(d):
            raise SystemExit(f"FATAL: {d} exists -- refusing to overwrite (version the name instead)")

    files = sorted(glob.glob(os.path.join(a.raw, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {a.raw}")
    print(f"{len(files)} shards")
    ds = load_dataset("parquet", data_files=files, split="train",
                      num_proc=a.num_proc, cache_dir=a.cache_dir)
    n = len(ds)
    print(ds)

    llama = 0
    for b in ds.select_columns(["token_count"]).with_format("numpy").iter(batch_size=100_000):
        llama += int(np.asarray(b["token_count"]).sum())
    print(f"exact Llama tokens: {llama / 1e9:.3f} B  ({llama / n:.0f}/doc)")

    # effective AFM tokens/doc on a random sample across all shards
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    idx = sorted(random.Random(a.seed).sample(range(n), min(a.sample_docs, n)))
    sample = list(ds.select(idx).select_columns(["text"])[:]["text"])
    afm = [len(x) for x in tok(sample, add_special_tokens=False)["input_ids"]]
    del sample
    K = effective(afm, a.seq_len)
    lost = sum(max(0, x + 1 - a.seq_len) for x in afm) / sum(x + 1 for x in afm)
    n1 = math.ceil(a.stage1_tokens * a.margin / K)
    n2 = math.ceil(a.stage2_tokens * a.margin / K)
    print(f"AFM tokens/doc (random {len(afm)}): {sum(afm) / len(afm):.1f}; truncated at {a.seq_len}: "
          f"{lost:.1%} of tokens; effective K = {K:.1f}")
    print(f"needed: stage1 {n1:,} docs (~{n1 * K / 1e6:.0f} M eff. tokens), "
          f"stage2 {n2:,} docs (~{n2 * K / 1e6:.0f} M), margin {a.margin}")

    ds = ds.select_columns(["text"])
    h = ds.map(_hash, batched=True, batch_size=10_000, num_proc=a.num_proc,
               remove_columns=["text"], desc="hashing")
    hashes = h[:]["h"]
    assert isinstance(hashes, list) and len(hashes) == n
    del h

    # ---- held-out: random positions; texts that also occur elsewhere are skipped
    rng = random.Random(a.seed + 1)
    cand = sorted(rng.sample(range(n), a.heldout_docs + a.heldout_spare))
    cand_set = set(cand)
    rest_hashes = {}
    for i, x in enumerate(hashes):
        if i not in cand_set:
            rest_hashes[x] = rest_hashes.get(x, 0) + 1
    held, held_h, leaked, dup_in_held = [], set(), 0, 0
    for i in cand:
        x = hashes[i]
        if x in rest_hashes:
            leaked += 1
            continue
        if x in held_h:
            dup_in_held += 1
            continue
        held_h.add(x)
        held.append(i)
        if len(held) == a.heldout_docs:
            break
    if len(held) < a.heldout_docs:
        print(f"!! WARNING: only {len(held)} of {a.heldout_docs} held-out docs (more leaks than "
              f"--heldout-spare {a.heldout_spare})")
    held_set = set(held)
    pool = [i for i in range(n) if i not in held_set and hashes[i] not in held_h]

    # ---- stage1 / stage2: one seeded permutation, distinct texts, disjoint by text
    random.Random(a.seed + 2).shuffle(pool)
    s1, s2, h1, h2, skip1, skip2, pos = [], [], set(), set(), 0, 0, 0
    while len(s1) < n1 and pos < len(pool):
        i = pool[pos]; pos += 1
        if hashes[i] in h1:
            skip1 += 1
            continue
        h1.add(hashes[i]); s1.append(i)
    while len(s2) < n2 and pos < len(pool):
        i = pool[pos]; pos += 1
        if hashes[i] in h1 or hashes[i] in h2:
            skip2 += 1
            continue
        h2.add(hashes[i]); s2.append(i)
    left = len(pool) - pos
    print(f"held-out {len(held):,} (skipped {leaked} whose text occurs elsewhere, {dup_in_held} repeats) · "
          f"stage1 {len(s1):,} (skipped {skip1} repeats) · stage2 {len(s2):,} (skipped {skip2} repeats / "
          f"texts in stage1) · unused {left:,}")
    if len(s1) < n1 or len(s2) < n2:
        short = (n1 - len(s1)) + (n2 - len(s2))
        raise SystemExit(f"FATAL: {short:,} docs short. Nothing was written. Download more: resubmit "
                         f"STEP=download with a larger MARGIN (it resumes, same revision, same shard "
                         f"order), then rerun prepare.")
    del hashes, rest_hashes, h1, h2, pool

    ds.select(held).save_to_disk(a.out + "_heldout")
    print(f"heldout {len(held):,} -> {a.out}_heldout")
    for suf, sel in (("_stage1", s1), ("_stage2", s2)):
        sel.sort()                                              # SOURCE order -> sequential writes
        ds.select(sel).save_to_disk(a.out + suf, num_proc=a.num_proc)
        print(f"{suf[1:]:7} {len(sel):,} -> {a.out + suf}")

    stats = {"raw": a.raw, "files": [os.path.relpath(f, a.raw) for f in files], "n_docs_total": n,
             "llama_tokens_exact": llama, "tokenizer": a.tokenizer, "seq_len": a.seq_len,
             "sample_docs": len(afm), "afm_tokens_per_doc": sum(afm) / len(afm),
             "effective_tokens_per_doc": K, "frac_tokens_truncated": lost, "margin": a.margin,
             "n_heldout": len(held), "heldout_skipped_leak": leaked, "heldout_skipped_repeat": dup_in_held,
             "n_stage1": len(s1), "stage1_tokens_target": a.stage1_tokens, "stage1_eff_tokens_est": len(s1) * K,
             "n_stage2": len(s2), "stage2_tokens_target": a.stage2_tokens, "stage2_eff_tokens_est": len(s2) * K,
             "skipped_repeat_stage1": skip1, "skipped_repeat_or_in_stage1_stage2": skip2,
             "unused_docs": left, "seed": a.seed}
    man = os.path.join(a.raw, "data_manifest.json")
    if os.path.exists(man):
        stats["download_manifest"] = json.load(open(man))
    with open(a.out + "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"stats -> {a.out}_stats.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("download", "prepare"):
        s = sub.add_parser(name)
        s.add_argument("--stage1-tokens", type=float, default=3e8)
        s.add_argument("--stage2-tokens", type=float, default=1e9)
        s.add_argument("--seq-len", type=int, default=2048)
        s.add_argument("--heldout-docs", type=int, default=2000)
        s.add_argument("--heldout-spare", type=int, default=200, help="extra candidates to replace leaks")
        s.add_argument("--seed", type=int, default=42)
        s.add_argument("--tokenizer", default=TOKENIZER)
    d = sub.choices["download"]
    d.add_argument("--out", required=True)
    d.add_argument("--margin", type=float, default=1.35,
                   help="download safety factor on the token target (> prepare's: the probe is one shard)")
    d.add_argument("--target-docs", type=int, default=0, help="override the computed doc target")
    d.add_argument("--revision", default=None, help="dataset commit; default: REVISION file, else current")
    d.add_argument("--probe-docs", type=int, default=2000)
    p = sub.choices["prepare"]
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True, help="base path; _heldout / _stage1 / _stage2 / _stats.json appended")
    p.add_argument("--margin", type=float, default=1.15, help="each split holds tokens x margin")
    p.add_argument("--sample-docs", type=int, default=5000)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--cache-dir", default=None, help="intermediate Arrow cache (deletable afterwards)")
    a = ap.parse_args()
    download(a) if a.cmd == "download" else prepare(a)


if __name__ == "__main__":      # REQUIRED: save_to_disk(num_proc) / map(num_proc) re-import this file
    main()