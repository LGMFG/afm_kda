"""Download a concrete amount of tokens from FineWeb-Edu (sample/100BT).

Low-memory variant of download_fineweb_tokens.py. Identical behaviour and
identical manifest, except that the calibration step streams the probe shard
instead of loading it whole. The original read all ~726k documents of the probe
shard into Python strings (~15 GB) just to sample 2000 of them, and held that
memory through the download -- which OOMs on lrd_all_serial (15.4 GB cap).
This version peaks at a few hundred MB.

What it does, in order:
  1. Pins the current dataset revision (commit hash).
  2. Streams the first --probe-docs documents of ONE probe shard and measures
     real tokens-per-doc with the given tokenizer (nominal "100BT" is GPT-2
     tokens, not AFM/Qwen tokens). Total row count comes from the parquet
     footer, so no extra data is read.
  3. Randomly selects (fixed seed) just enough shards to cover the target.
  4. Downloads only those shards.
  5. Writes data_manifest.json (revision, seed, shard list) next to the data,
     which makes the subset exactly reproducible.

Note vs. the original: the probe sample is the FIRST --probe-docs documents of
the shard rather than a random sample of all of them. FineWeb shards are not
ordered by length, and the target is over-provisioned anyway, so the resulting
estimate is equivalent in practice.

Usage (node with internet, e.g. lrd_all_serial on Leonardo):
    pip install -U transformers huggingface_hub pyarrow
    hf auth login   # optional: higher rate limits, faster downloads
    python download_fineweb_tokens_lowmem.py \
        --target-tokens 3e9 \
        --tokenizer arcee-ai/AFM-4.5B-Base \
        --out-dir $WORK/WORK_LUKAS/datasets/fineweb_edu_raw

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
                    help="Documents read from the probe shard to calibrate")
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
    #    Streamed: only --probe-docs documents are ever held in memory.
    print("downloading one probe shard for calibration ...")
    probe = hf_hub_download(REPO, files[0], repo_type="dataset", revision=rev)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    pf = pq.ParquetFile(probe)
    n_rows = pf.metadata.num_rows              # from the footer; reads no text
    sample = []
    for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
        sample.extend(batch.column("text").to_pylist())
        if len(sample) >= args.probe_docs:
            break                              # rest of the shard untouched
    sample = sample[: args.probe_docs]

    ids = tok(sample, add_special_tokens=False)["input_ids"]
    per_doc = sum(len(x) for x in ids) / len(sample)   # +1 EOS/doc is noise
    shard_tokens = per_doc * n_rows
    del sample, ids, pf                        # free before the download
    print(f"calibration: {per_doc:.0f} tokens/doc -> "
          f"~{shard_tokens/1e9:.3f}B {args.tokenizer} tokens per shard "
          f"({n_rows:,} docs/shard)")

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
        "docs_per_shard": n_rows,
        "tokens_per_doc_estimate": per_doc,
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