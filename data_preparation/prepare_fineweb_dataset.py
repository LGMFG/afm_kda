"""Split downloaded FineWeb-Edu parquet shards into train / held-out
save_to_disk datasets for DistillKit, and report the real token count.

NOTE ON PURPOSE: DistillKit does NOT need this for tokenization. TRL's
SFTTrainer tokenizes and packs internally from the `text` column
(see main.py: processing_class=tokenizer, max_length=sequence_length).
This script exists for two reasons only:
  1. To carve off a held-out slice that is guaranteed never to be trained on,
     which DistillKit will not do for you and which the held-out-KL metric
     requires.
  2. To produce a defensible token count for the thesis.
It also drops the nine unused FineWeb columns, which shrinks the prepared
dataset cache.

Produces:
    <out-dir>_train      training set     -> DistillKit `kind: local` disk_path
    <out-dir>_heldout    held-out slice   -> never trained on
    <out-dir>_stats.json token/row counts

Token accounting, two independent ways:
  * exact GPT-2 tokens, summed from FineWeb-Edu's own `token_count` column
  * AFM tokens, from an unbiased RANDOM sample across all downloaded shards
    (the download script's probe reads the head of one shard, which runs ~9%
    long -- 1080 vs 993 tokens/doc)

Usage (lrd_all_serial: budget-free, 4 cores / 15.4 GB / 4 h):
    python prepare_fineweb_dataset.py \
        --raw-dir  $WORK_LUKAS/datasets/fineweb_edu_raw \
        --out-dir  $WORK_LUKAS/datasets/fineweb_edu_3B \
        --num-proc 4
"""

import argparse
import glob
import json
import os
import random

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", required=True,
                    help="Output dir of the download script")
    ap.add_argument("--out-dir", required=True,
                    help="Base path; _train / _heldout / _stats.json appended")
    ap.add_argument("--tokenizer", default="arcee-ai/AFM-4.5B-Base")
    ap.add_argument("--heldout-docs", type=int, default=2000)
    ap.add_argument("--sample-docs", type=int, default=5000,
                    help="Documents tokenized for the AFM token estimate")
    ap.add_argument("--num-proc", type=int, default=4,
                    help="Match your allocated cores; 4 on lrd_all_serial")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default=None,
                    help="Intermediate Arrow cache (~13 GB, deletable after)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw_dir, "**", "*.parquet"),
                             recursive=True))
    if not files:
        raise SystemExit(f"no parquet files under {args.raw_dir}")
    print(f"{len(files)} shards:")
    for f in files:
        print("   ", os.path.basename(f))

    ds = load_dataset("parquet", data_files=files, split="train",
                      num_proc=args.num_proc, cache_dir=args.cache_dir)
    print(ds)
    n_docs = len(ds)

    # ---- exact GPT-2 token count, from the dataset's own column ------------
    # Batched: datasets 4.x returns a lazy Column from ds[name], and we don't
    # want the whole column resident anyway.
    gpt2_tokens = None
    if "token_count" in ds.column_names:
        total = 0
        for batch in (ds.select_columns(["token_count"])
                        .with_format("numpy")
                        .iter(batch_size=100_000)):
            total += int(np.asarray(batch["token_count"]).sum())
        gpt2_tokens = total
        print(f"exact GPT-2 tokens (token_count column): "
              f"{gpt2_tokens/1e9:.3f}B")
    else:
        print("no token_count column found; skipping exact GPT-2 count")

    # ---- unbiased AFM token estimate, random sample of the real shards -----
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    sample_idx = random.Random(args.seed).sample(
        range(n_docs), min(args.sample_docs, n_docs))
    # list(...) because ds[col] is a lazy Column in datasets 4.x
    sample = list(ds.select(sample_idx)["text"])
    ids = tok(sample, add_special_tokens=False)["input_ids"]
    afm_per_doc = sum(len(x) for x in ids) / len(sample)
    afm_tokens = afm_per_doc * n_docs
    print(f"AFM tokens/doc (random sample of {len(sample)}): "
          f"{afm_per_doc:.1f}")
    print(f"estimated AFM tokens total: {afm_tokens/1e9:.3f}B")
    del sample, ids

    # ---- keep only what training needs ------------------------------------
    ds = ds.select_columns(["text"])

    # ---- held-out split ---------------------------------------------------
    # Random held-out indices (representative), but the train indices stay in
    # ascending order so save_to_disk writes sequentially instead of shuffling
    # ~13 GB into random order on a parallel filesystem.
    rng = random.Random(args.seed + 1)
    held_idx = sorted(rng.sample(range(n_docs), args.heldout_docs))
    held_set = set(held_idx)
    train_idx = [i for i in range(n_docs) if i not in held_set]

    train_dir = args.out_dir + "_train"
    held_dir = args.out_dir + "_heldout"

    ds.select(held_idx).save_to_disk(held_dir)
    ds.select(train_idx).save_to_disk(train_dir, num_proc=args.num_proc)
    print(f"train:   {len(train_idx):,} docs -> {train_dir}")
    print(f"heldout: {len(held_idx):,} docs -> {held_dir}")

    # ---- freeze the numbers -----------------------------------------------
    stats = {
        "raw_dir": args.raw_dir,
        "n_shards": len(files),
        "shards": [os.path.basename(f) for f in files],
        "n_docs_total": n_docs,
        "n_docs_train": len(train_idx),
        "n_docs_heldout": len(held_idx),
        "tokenizer": args.tokenizer,
        "afm_tokens_per_doc": afm_per_doc,
        "afm_tokens_total_estimate": afm_tokens,
        "gpt2_tokens_exact": gpt2_tokens,
        "sample_docs_for_estimate": len(sample_idx),
        "seed": args.seed,
    }
    stats_path = args.out_dir + "_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"stats written to {stats_path}")


if __name__ == "__main__":
    main()