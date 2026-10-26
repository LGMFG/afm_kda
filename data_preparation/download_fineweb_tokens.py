"""Download a concrete amount of tokens from FineWeb-Edu (sample/100BT).

What it does, in order:
  1. Pins the current dataset revision (commit hash).
  2. Downloads ONE probe shard and measures real tokens-per-shard with the
     given tokenizer (nominal "100BT" is GPT-2 tokens, not AFM/Qwen tokens).
  3. Randomly selects (fixed seed) just enough shards to cover the target.
  4. Downloads only those shards.
  5. Writes data_manifest.json (revision, seed, shard list) next to the data,
     which makes the subset exactly reproducible.

Usage (node with internet, e.g. lrd_all_serial on Leonardo):
    pip install -U transformers huggingface_hub pyarrow
    hf auth login   # account must have accepted arcee-ai/AFM-4.5B-Base terms
    python download_fineweb_tokens.py \
        --target-tokens 3e9 \
        --tokenizer arcee-ai/AFM-4.5B-Base \
        --out-dir $CINECA_SCRATCH/fineweb-edu-subset

Interrupted runs can simply be re-launched: already-downloaded shards are
skipped automatically.
"""

import argparse
import json
import math
import os
import random

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import AutoTokenizer

REPO = "HuggingFaceFW/fineweb-edu"
PREFIX = "sample/100BT/"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-tokens", type=float, required=True,
                    help="How many tokens to download, e.g. 3e9")
    ap.add_argument("--tokenizer", default="arcee-ai/AFM-4.5B-Base",
                    help="Tokenizer that defines what a 'token' is")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for shard selection (part of the manifest)")
    ap.add_argument("--probe-docs", type=int, default=2000,
                    help="Documents sampled from the probe shard to calibrate")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    api = HfApi()

    # 1. Pin the revision so the subset stays reproducible forever.
    rev = api.repo_info(REPO, repo_type="dataset").sha
    files = sorted(
        f for f in api.list_repo_files(REPO, repo_type="dataset", revision=rev)
        if f.startswith(PREFIX) and f.endswith(".parquet")
    )
    print(f"revision {rev[:12]}: {len(files)} shards under {PREFIX}")

    # 2. Calibrate: real tokens per shard, measured with the target tokenizer.
    print("downloading one probe shard for calibration ...")
    probe = hf_hub_download(REPO, files[0], repo_type="dataset", revision=rev)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    table = pq.read_table(probe, columns=["text"])
    docs = table["text"].to_pylist()
    sample = random.Random(0).sample(docs, min(args.probe_docs, len(docs)))
    ids = tok(sample, add_special_tokens=False)["input_ids"]
    per_doc = sum(len(x) for x in ids) / len(sample)  # +1 EOS/doc is noise
    shard_tokens = per_doc * table.num_rows
    print(f"calibration: {per_doc:.0f} tokens/doc -> "
          f"~{shard_tokens/1e9:.3f}B {args.tokenizer} tokens per shard")

    # 3. Select just enough shards, deterministically.
    n_shards = max(1, math.ceil(args.target_tokens / shard_tokens))
    if n_shards > len(files):
        raise SystemExit(f"target {args.target_tokens:.2e} exceeds the "
                         f"~100BT sample; choose a smaller target")
    chosen = sorted(random.Random(args.seed).sample(files, n_shards))
    est = n_shards * shard_tokens
    print(f"selected {n_shards} shards "
          f"(~{est/1e9:.2f}B tokens for target {args.target_tokens:.2e})")

    # 4. Download only those shards (resumable; re-run to continue).
    snapshot_download(REPO, repo_type="dataset", revision=rev,
                      allow_patterns=chosen, local_dir=args.out_dir)

    # 5. Freeze the manifest.
    manifest = {
        "repo": REPO,
        "revision": rev,
        "subset_prefix": PREFIX,
        "selection_seed": args.seed,
        "tokenizer_for_budget": args.tokenizer,
        "tokens_per_shard_estimate": shard_tokens,
        "target_tokens": args.target_tokens,
        "estimated_total_tokens": est,
        "n_shards": n_shards,
        "shards": chosen,
    }
    path = os.path.join(args.out_dir, "data_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"done. manifest written to {path}")


if __name__ == "__main__":
    main()
