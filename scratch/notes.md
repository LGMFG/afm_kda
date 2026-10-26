2026-09-09 — AFM-4.5B-Base config inconsistency
transformers warns: explicit YaRN rope factor = 20.0, but implicit factor from
max_position_embeddings / original_max_position_embeddings = 16.0. transformers
uses 20.0. Inert at seq_len <= 4096. REVISIT if long-context (stage 3) results
look wrong.

2026-09-09 — FineWeb-Edu subset: 4 shards, revision 87f09149ef47, seed 42.
Manifest says ~3.14B tokens, but that estimate came from the first 2000 docs
of the probe shard, which run long. Unbiased estimate from a random sample of
the full shard is 993 tok/doc → ~2.88B tokens actually available. Use 2.9B.

2026-09-10 — FineWeb-Edu subset, final numbers
  revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, seed 42, 4 of 140 shards
  2,923,000 docs total | 2,921,000 train | 2,000 held-out
  3.015B GPT-2 tokens (exact, from token_count column)
  ~2.83B AFM tokens (970.1 tok/doc, random sample of 5000)
  Manifest's 3.14B is ~11% high (probe read the head of one shard) — use 2.83B.