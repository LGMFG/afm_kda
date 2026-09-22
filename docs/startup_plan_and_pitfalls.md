# Thesis Startup Plan — Architecture Distillation (AFM-4.5B + KDA + DistillKit)

*Part 1 = the startup phase, step by step. Part 2 = things that are easy to overlook. Part 3 = multi-node training. Part 4 = Leonardo facts from the CINECA docs, verified.*

*Revision 18 (2026-09-18) — **the startup phase is over; R1 is submitted.** Step 8's estimates are replaced by R0's **measurements** (7,585 tok/s at 1 node → ~22,800 at 4). **8b and 8c are downgraded** to optional, with reasons. Three hard-won pitfalls added to Part 2: `min_lr` under DeepSpeed, `prepared_dataset_path` at scale, and `functionary_packing` needing `dataset.prepacked`. Structure unchanged — this document stays a reference, not a live log.*

*Revision 17 — final structure: steps 1–7 closed, step 7 = building the student, step 8 = training infrastructure. Revision 16 — the corpus decision recorded with its rationale.*

**The live document is `claude/first_distillkit_runs.md`.** Companions: `claude/building_the_student.md` (the model), `claude/sbatch_explained.md`, `claude/multinode_debug_notes.md`.

---

# PART 1 — The startup phase

## ⭐ What each step established

| Step | What it had to settle | What was done | How we knew it worked |
|---|---|---|---|
| **0** | Is there budget and somewhere to put things? | Read `saldo -b`; found `~/.bash_profile` had never had the exports added and appended them | ~3,404 Booster node-hours left; `$HF_HOME` stopped filling `$HOME` |
| **1** | Get a corpus onto Leonardo without a GPU | Rewrote the supervisor's download script to stream parquet instead of loading whole shards, ran it on the budget-free serial partition | 4 shards, 8.6 G, matching `data_manifest.json` exactly |
| **2** | **The project's biggest unknown:** does KDA even run on A100? fla tests only H100 | Installed `flash-linear-attention[cuda]` 0.5.2, ran forward+backward at AFM's real width | **480,458 tok/s at 2560**, every parameter gets a gradient — a fallback path would have looked like 2% of that |
| **3** | A corpus we can cite, and a held-out split carved before any training | Prepared the Arrow dataset; measured tokens per document two ways | **3.015 B GPT-2 tokens**, **774 effective tokens/doc** after packing |
| **4** | Fit a 4.6 B model's training state on a 64 GB card | DeepSpeed ZeRO-2: shard Adam state and gradients, replicate weights | 74 GB/GPU naive → ~25 GB at 4 GPUs. Later confirmed exactly by the memory model |
| **5** | Does DistillKit pair teacher and student correctly? | Distilled AFM into **itself** — the one configuration with a known answer | **KL exactly 0** and `hs_cosine` 3.2e-5 at step 0. Also settled `layer_mapping: all` by reading source, since self-distillation cannot test it |
| **6** | One node is too slow — does four work? | Fixed the launcher (`--deepspeed_multinode_launcher standard`), then measured strong *and* weak scaling | **75.3% efficiency at 4 nodes**, 3.01× speedup; memory model predicted 12.72 GB and measured 12.721 |
| **7** | Build the KDA student, and prove it is what we think | Subclassed the library's Arcee code rather than copying Arcee's; verified against the teacher at every step → **`claude/building_the_student.md`** | All-attention student **bit-identical** to the teacher; hybrid at **4.9990 B** exactly as predicted |
| **8** | Stop the first long run from wasting node-hours | Token budget from a measurement, not an estimate; the rest downgraded once R0 made them cheap | ✅ **R1 submitted 2026-09-18** |

**Two habits did most of the work.** Every step was verified against something with a known answer — self-distillation for the trainer, the unmodified teacher for the student, a fitted two-point model for memory. And every suspicious result was run twice, which is how a non-deterministic initialization bug and a silently-split multi-node job were both caught.

**A third habit, learned the hard way in step 8:** *read how a configuration flag is consumed before building on what its name implies.* See Part 2.

---

## Step 0 — Budget ✅ / Storage ✅

`saldo -b` on 2026-09-08 for `EUHPC_D34_049`:

| | local hours | Booster node-hours (4×A100) |
|---|---|---|
| Total grant (2026-03-05 → 2027-03-05) | 144,000 | 4,500 |
| Consumed | 35,087 (24.4%) | 1,097 |
| **Remaining** | **108,913** | **~3,404** |
| Monthly quota | 11,835 | ~370 |

**Conversion:** BH = T · N · R · C, where R is the largest fraction of the node occupied and C = 32 cores. All 4 GPUs → R = 1.0, so **1 node-hour = 32 local hours**. Multi-node multiplies by N.

**Caveat:** the ~370 node-hours/month is a *priority* mechanism ("budget linearization"), not a hard stop.

### ✅ The `$HF_HOME` mystery — solved and fixed

`~/.bash_profile` **does** source `~/.bashrc`, so the bridge was never missing. The `ls -la` timestamp gave it away — **Apr 13 2023, 376 bytes**, the untouched file CINECA created with the account. The exports had never been added. Now appended:

```bash
export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS
export HF_HOME=$WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PIP_CACHE_DIR=$WORK_LUKAS/.cache/pip
export XDG_CACHE_HOME=$WORK_LUKAS/.cache
```

**Bash file rules.** A **login** shell (SSH) reads `~/.bash_profile`, or `~/.bash_login`, or `~/.profile` — the first that exists, and stops. An **interactive non-login** shell reads `~/.bashrc`. A **SLURM batch script is neither** and reads no startup file at all, which is why every job must export its own variables.

---

## Step 1 — FineWeb-Edu download ✅ DONE

Ran on `lrd_all_serial` (budget-free), job 57007612, using `claude/download_fineweb_tokens_lowmem.py`.

**Why the low-memory rewrite was needed:** the supervisor's original script calibrates tokens-per-document by loading an entire parquet shard into memory (~726k documents, ~15 GB) and holds it in scope during the download. `lrd_all_serial` caps memory at 15400 MB tied to the 4-core allocation, and `--mem=32G` is refused with `QOSMaxMemoryPerUser`. The fix streams with `pq.ParquetFile(...).iter_batches()` and takes the row count from the file footer. Peak dropped from ~15 GB to a few hundred MB.

**Result:** 4 shards, 8.6 G of parquet, matching `data_manifest.json` exactly.

```
000_00006.parquet   002_00008.parquet   006_00002.parquet   007_00000.parquet
```

Manifest: revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, seed 42. **Keep it in git.**

⚠️ **The manifest's `estimated_total_tokens` of 3.137e9 is an overestimate — do not quote it.** See step 3.

---

## Step 2 — flash-linear-attention + KDA on A100 ✅ DONE

**This was the highest-risk unknown in the project**, because fla's CI covers only H100 and its benchmarks are on GB200 — nothing tests A100 (sm80). It works, and it is not slow.

### 2a. Install — clean

```bash
pip install 'flash-linear-attention[cuda]'
```

Added exactly two packages; `torch 2.12.0+cu126` and `triton 3.7.0` unchanged:

```
> fla-core==0.5.2
> flash-linear-attention==0.5.2
```

fla v0.5+ requires a **backend extra** (`[cuda]`) and no longer transitively installs torch/triton — which is exactly why this install was safe.

⚠️ **Pinned to 0.5.2, and the pin is load-bearing.** 0.5.2 overwrites a supplied `cu_seqlens` whenever an `attention_mask` is also present; GitHub's default branch does not. Record the fla version in every run log. `kda_installed_0.5.2.py` in the project folder is the authoritative copy — **not** anything downloaded from GitHub.

### 2b. The real API, recorded (fla 0.5.2)

`fla.layers` exposes: `DeltaFormerAttention`, `DeltaNet`, `GatedDeltaNet`, `GatedDeltaNet2`, `GatedDeltaProduct`, `KimiDeltaAttention`.

```python
KimiDeltaAttention.__init__(
    self, hidden_size=2048, expand_v=1, head_dim=128, num_heads=16,
    num_v_heads=None, mode='chunk', use_short_conv=True,
    allow_neg_eigval=False, safe_gate=False, lower_bound=None,
    conv_size=4, conv_bias=False, layer_idx=None, norm_eps=1e-05, **kwargs,
)
```

⚠️ **Leave `num_v_heads` at `None`.** Arcee's blog post: *"You can set num_v_heads, but that doesn't do anything for your key heads and I ran into some crashes when training with it enabled anyway."*

⚠️ **Never splat a config dict into this constructor.** The trailing `**kwargs` swallows misspellings silently.

### 2c–2d. Correctness and throughput — passed

`claude/test_kda.py`, bf16, `mode='chunk'`, one A100. All parameters receive gradients — the check that matters, since a layer that forwards fine but silently fails to train some parameters costs weeks.

| | hidden 2048 (16×128) | **hidden 2560 (20×128) — AFM's real shape** |
|---|---|---|
| tokens/s (B=1, T=4096, fwd+bwd) | 591,795 | **480,458** |
| peak memory | 0.7 GB | **0.9 GB** |

**1.23× slower for a 1.25× wider model** — proportional, so nothing pathological happens at the real dimensions. **The significance is diagnostic:** a broken kernel path looks like 2% of peak because it fell back to something generic. This didn't.

✅ **The comparison against full attention is done — step 7 (A6).** KDA is **0.31× at 2048**, crosses over at ~16k, **1.71× at 32k**. ⚠️ At the *model* level the penalty is only ~12% (R0), because the MLPs dilute it.

---

## Step 3 — Dataset preparation ✅ DONE

### ⚠️ Why FineWeb-Edu and not Arcee's corpus

Arcee's blog post trains on `arcee-train/afm-autodistill-mix-v0`, an **Arcee-internal instruction-style distillation mix**. It is not part of your pipeline, so the corpus had to be built independently — hence the whole FineWeb-Edu download effort in step 1, from Anamaria's script.

**FineWeb-Edu is the better choice for a thesis anyway**: public, documented, versioned by revision hash, and citable, so the data pipeline is reproducible by a reader. Arcee's mix is none of those things.

**The consequence, and it must be stated in the thesis:** your loss numbers, held-out KL, and benchmark scores are **not directly comparable to the blog post's**, because the training distribution differs. Arcee's figures (KL ≈ 0.2 on ~300 M tokens) are useful as *order-of-magnitude targets*, not as baselines to beat.

### The numbers to use

| Quantity | Value | Provenance |
|---|---|---|
| Documents (FineWeb-Edu `sample/100BT`, 4 shards) | 2,923,000 | exact |
| GPT-2 tokens | **3.015 B** | exact — FineWeb's own `token_count` column |
| AFM tokens (raw tokenizer) | ~2.84 B | estimated, random 5,000-doc sample → 970.1 tok/doc |
| **Effective tokens after packing** | **~774 tok/doc → ~2.26 B** | 5,000 docs → 1,889 packed sequences of 2048 |
| Train split | 2,921,000 docs | `$WORK_LUKAS/datasets/fineweb_edu_3B_train` — **verified** |
| Held-out split | 2,000 docs | `..._heldout`, never trained on |

**Use 774 tok/doc for budgeting, 3.015 B GPT-2 tokens for citing the corpus.** One epoch is **~2.26 B usable tokens**.

**Why this script existed at all:** not for tokenization — **DistillKit does not need pre-tokenized data.** `_format_row` only fires when neither `text` nor `input_ids` exists, and FineWeb has `text`. Its real value is (a) the held-out split, carved off before any training, and (b) a defensible token count.

**On the held-out split:** at ~1.3B tokens against a 2.26B effective corpus you consume ~58% of one epoch, so train and held-out will be nearly identical. **Every run confirms this** (1.097 vs 1.083; 0.4394 vs 0.4339; 0.4331 vs 0.4337). Cheap insurance and a defensible sentence in the thesis, not a correctness requirement.

---

## Step 4 — DeepSpeed ✅ DONE

```
Successfully installed deepspeed-0.19.6 hjson-3.1.0 msgpack-1.2.2 py-cpuinfo-9.0.0
torch 2.12.0+cu126 12.6   triton 3.7.0     ← unchanged
```

### What DeepSpeed and accelerate actually do

**accelerate is only a launcher.** It starts the right number of Python processes, tells each which GPU it owns and what its rank is, and connects them. It does no memory work.

**DeepSpeed solves a memory problem.** Training needs ~16 bytes per parameter: 2 for the bf16 weight, 2 for its gradient, 12 for Adam's bookkeeping. At 4.62B parameters that is **74 GB** — more than a 64 GB A100. Ordinary data-parallel training puts a *complete copy* on every GPU.

**ZeRO** shards those per-parameter costs across GPUs instead of replicating them. Stage 1 splits Adam's state; **stage 2 splits Adam's state and the gradients** (what you use); stage 3 also splits the weights. At stage 2 on 4 GPUs: 14 bytes of gradient+optimizer divide four ways (3.5 bytes), the 2-byte weight stays everywhere → 5.5 bytes/param ≈ 25 GB. It fits.

Your own measurements *are* this equation: `alloc_delta = 9.25 + 55.46/W` GB. The price is communication — the 12–25% lost to scaling inefficiency in step 6.

⚠️ **Stage 1 uses much less.** Once `frozen_res` applies, only ~945 M parameters (19%) carry Adam state — R0 measured 12.86 GB at 4 GPUs against the all-trainable fit's 23.12.

### Runtime warnings — harmless

`Unable to run /usr/local/cuda/bin/nvcc` for `fp_quantizer`, `transformer_inference`, `inference_core_ops`, `cutlass_ops`, `ragged_device_ops`, `ragged_ops`, `spatial_inference`. **Every one is a DeepSpeed-*Inference* op**, unused by ZeRO-2 training.

⚠️ **JIT compilation needs a per-job extension cache.** Default is `~/.cache/torch_extensions` on a shared filesystem, where ranks race. Set `TORCH_EXTENSIONS_DIR` in every job.

### The config — `$WORK_LUKAS/distillkit/ds_config_zero2.json`

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2, "overlap_comm": true, "contiguous_gradients": true,
    "reduce_bucket_size": 5e8, "allgather_bucket_size": 5e8
  },
  "gradient_accumulation_steps": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

`"auto"` values resolve from the HF `TrainingArguments`, so set batch sizes in the **YAML**, not here. ⚠️ **And DeepSpeed's resolved value wins** if they ever disagree — the `Gradient accumulation steps mismatch … Using DeepSpeed's value` warning is normal.

**The teacher is not sharded.** `trainer.py` does `teacher_model.to(self.accelerator.device)` — a full bf16 replica on every GPU, outside DeepSpeed's control. ~9.2 GB per GPU, and more nodes will not shrink it.

⚠️ **`reduce_bucket_size` / `allgather_bucket_size` at 5e8 are counts of *elements*, not bytes.** Dropping both to `2e8` is the cheapest lever if you need headroom.

---

## Step 5 — Self-distillation smoke tests ✅ **BOTH PASSED**

Runs **57223076** (cross-entropy + KL) and **57229133** (+ `hs_cosine`, `layer_mapping: all`).

### 5a. Model cache — verified complete (after a real failure)

The cached directory existed but its snapshot held only the tokenizer and config — **no weight file**. `HF_HUB_OFFLINE=1` correctly refused to fetch the rest. Fixed by re-running the load without the offline flag on a login node, then re-running with it to prove the compute node would succeed.

**General pattern:** to verify a cache before a real job, attempt the offline load exactly as the job will. "The directory exists" is not proof. And **always prefix with `HF_HOME=$WORK_LUKAS/hf_cache`**.

⚠️ **Remote-code models have a second cache.** `$HF_HOME/modules/transformers_modules/<name>` holds copies of the student's `.py` files. It must be deleted after any `.py` change or rename, and a job must pre-warm it once in a single process before `srun`.

### AFM-4.5B-Base's real architecture

| Quantity | Value | Why it matters |
|---|---|---|
| `num_hidden_layers` | **36** | `layer_mapping: all` builds 36 pairs |
| `len(hidden_states)` | **37** | index 0 is the embedding output |
| `hidden_size` | **2560** | **not** the 2048 in the first KDA tests |
| `num_attention_heads` / `num_key_value_heads` | **20 / 4** | → the 5× GQA expansion for KDA |
| `head_dim` | **128** | |
| `tie_word_embeddings` | **False** | sidesteps RADLADS's tied-embedding workaround |
| total parameters | **4.6192 B** | baseline for GQA-expansion growth |

### 5b–5d. The config and the results

The full `smoke_test.yaml` is superseded by `r1_stage1_4node.yaml` in the runs doc; what matters here is what it proved:

```
{'distillation_loss/1_cross_entropy': '2.226', 'distillation_loss/2_kl': '0', 'epoch': 0}
{'distillation_loss/3_hs_cosine': '3.172e-05'}
```

**KL is exactly zero on the first forward pass** — identical models, identical distributions. The strongest available confirmation that DistillKit pairs teacher and student correctly. **`hs_cosine` is 3.2e-5** — zero to within bf16 rounding (cosine goes through dot products and norms that accumulate error, unlike KL's exact `log(p/p)`).

Steady state **~0.7 s/step** (CE+KL), **~0.85 s/step** with `hs_cosine` — the hidden-state loss shape costs **~20% more per step**.

⚠️ **The 2.226 is the *teacher* distilling into itself.** The KDA student's step-0 cross-entropy is ≈ **7.94**. R0 measured **7.918**. If a KDA run ever shows ~2.2, the wrong model loaded.

**On `report_to`.** `none` removes a moving part — every metric goes to the `.out` file. `wandb` buys loss curves and a link to send Anamaria; it needs a free account, a one-time `wandb login` on a **login** node, `WANDB_MODE=offline` in the job, then `wandb sync $WORK_LUKAS/wandb/wandb/offline-run-*`. HF's default is `report_to: "all"`, which means wandb whenever installed — so set it explicitly either way.

### 5e. Three real problems found

**⚠️ Problem 1 — dataset preparation will not scale.** `Saving the dataset: 5000 [01:46, 47.03 examples/s]`. `_load_dataset` shuffles all 2.92M documents then selects, so `save_to_disk` performs scattered random reads across a 20 GB memory-mapped Arrow file on Lustre, and DistillKit calls it with **no `num_proc`**. ✅ **Resolved the simple way: delete `prepared_dataset_path` from the config.** At R1's 500,000 documents the cache would have cost ~2 hours *inside the training job*; without it, TRL tokenizes at ~2,700 docs/s with `num_proc: 16` — about 3 minutes. The cache was a false economy at scale.

**⚠️ Problem 2 — DistillKit always writes a full model.** `do_distill` calls `trainer.save_model(config.output_path)` unconditionally. **`save_strategy: "no"` does not prevent it.** It writes to `output_path` — mind which directory that is — and also drops a `distillkit_config.yaml` snapshot there at startup, which is the authoritative record of what a run actually used.

**⚠️ Problem 3 — eval is inside `train_runtime`.** The first 1-node scaling run spent 23.78 s of its 148 s on four eval passes; comparing that against a 4-node run with eval disabled inflated apparent efficiency from 75% to 91%. **Always set `eval_strategy: "no"` for any run whose timing you quote**, and check with `grep -c "eval_loss"`.

**Warnings confirmed harmless:** the `nvcc` wall; `Detected kernel version 4.18.0`; the YaRN factor mismatch (AFM's own config inconsistency, and doubly moot since the hybrid runs `use_rope: false`); `Token indices sequence length is longer than the specified maximum` (the tokenizer commenting before the packer cuts rows). **Not harmless:** a **Triton** CPU-fallback message inside a job.

### 5f. ✅ The layer mapping is CORRECT — keep `layer_mapping: all`

`len(hidden_states) == 37` for 36 layers, and `hidden_states[0]` is the embedding output. `main.py` builds `[(i, i) for i in range(36)]` → indices 0–35, and `lossfuncs/hidden_state.py` indexes bare.

Write the chain: `h0 = embeddings(tokens)`, `h1 = Layer0(h0)`, …, `h36 = Layer35(h35)`. Indices 0–35 are exactly the **inputs** to layers 0–35, so `all` aligns *every layer's input*. The alternative (1–36) would align *every layer's output*. Both are 36 comparisons sharing `h1`–`h35`, differing in one term — and **both candidate terms are duds**:

- `h0` (included) is **identically zero forever**: `frozen_res` freezes `embed_tokens` and the student starts from the teacher.
- `h36` (the alternative) is **already pinned by KL**: `norm.weight` and `lm_head` are frozen and identical, RMSNorm depends only on direction, and cosine measures only direction.

35 informative constraints plus one dead term either way. **Flag if any of these change:** `lm_head` or the final norm unfrozen, `hs_cosine` without a KL term, or a student whose depth differs from the teacher's.

**Self-distillation cannot test this** — with identical models *any* mapping gives cosine 1. This needed source reading, not an experiment.

⚠️ **`_can_record_outputs` must name the student's own layer class**, or `output_hidden_states=True` returns nothing and `hs_cosine` silently compares nothing.

---

## Step 6 — Multi-node ✅ **DONE — fully characterized**

*Debugging record in `claude/multinode_debug_notes.md`; line-by-line sbatch walkthrough in `claude/sbatch_explained.md`.*

### The fix

`accelerate launch --use_deepspeed` with `--num_machines > 1` hands off to DeepSpeed's own launcher, which without a hostfile builds a world from the local node only. Add **`--deepspeed_multinode_launcher standard`**, and rename `WORLD_SIZE` → `TOTAL_PROCS` (torch.distributed reads `WORLD_SIZE` from the environment itself).

### The measurements

All runs: AFM-4.5B self-distillation, CE+KL, `sequence_length 2048`, `per_device 1`, `max_steps 20`, **`eval_strategy: "no"`**, ZeRO-2.

**Strong scaling — global batch held at 32.** *Does the same job finish faster?*

| Nodes | GPUs | grad_accum | job | runtime | seq/s | tokens/s | speedup | **efficiency** |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 8 | 57329824 | 122.2 s | 5.238 | 10,727 | 1.00× | 100% |
| 2 | 8 | 4 | 57337956 | 69.16 s | 9.254 | 18,952 | 1.77× | **88.3%** |
| 4 | 16 | 2 | — | 40.57 s | 15.77 | 32,308 | 3.01× | **75.3%** |

**Weak scaling — `grad_accum: 8` held fixed.** *Does each GPU keep pulling its weight?*

| Nodes | GPUs | global batch | job | runtime | seq/s | seq/s per GPU | **efficiency** |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 32 | 57329824 | 122.2 s | 5.238 | 1.310 | 100% |
| 2 | 8 | 64 | 57329797 | 131.7 s | 9.719 | 1.215 | **92.8%** |
| 4 | 16 | 128 | 57337271 | 142.6 s | 17.95 | 1.122 | **85.7%** |

**How to read them together.** Weak scaling is always the flatter curve, and the gap between the series *is* the communication cost: at `grad_accum 8` each GPU runs eight forward/backward passes before gradients are exchanged; at 2 it runs two, so the same all-reduce is amortized over a quarter of the compute. **Quote the strong-scaling numbers** when justifying node count — they answer the question Anamaria asked — and keep weak scaling as supporting evidence that the loss is communication, not a defect.

⚠️ **Footnote on the 4-node weak-scaling row.** 20 steps × 128 = 2,560 sequences, but `num_samples: 5000` yielded only ~1,889, so it wrapped into a second epoch over page-cached data. 142.6 s is slightly optimistic and 85.7% is a mild upper bound.

**Design lesson:** `grad_accum` lives in the YAML and node count in the sbatch, which is how the first 2-node re-run silently kept `grad_accum: 8` and looked slower than one node while doing twice the work. **Keep one YAML per node count.** The same trap in sbatch scripts generally: **a variable assigned inside the script overrides `--export`** — write `VAR=${VAR:-default}` if the submission should win.

### The memory model — fitted, then confirmed to 0.03%

```
A + B/4  = 23.116 GB          A = 9.25 GB   = 4.62e9 × 2 B  (replicated bf16 weights)
A + B/8  = 16.184 GB    =>    B = 55.46 GB  = 4.62e9 × 12 B (Adam fp32 state, sharded)
```

Fitted on the 1- and 2-node points, this predicted **12.72 GB** at 16 GPUs. Measured **12.721 GB** — a 4 MB error.

**`alloc_delta` does not depend on `grad_accum`** — the 2-node runs at 8 and 4 returned byte-identical 16,184,188,416. Memory is a function of world size alone. (Contrast the broken runs, where `alloc_delta` was byte-identical across *different node counts* — that was the smoking gun.)

⚠️ **The constants are for the 4.6192 B teacher with everything trainable.** The frozen student is a different fit — R0 measured 12.86 GB at 4 GPUs where this model predicts 23.12. Refit rather than reusing it as a pass/fail test.

### Recommendation: **4 nodes**

At 32,308 tokens/s a 1.3B-token run takes **11.2 hours** instead of 34.2, for ~1,430 local hours against ~1,095 — about **30% more budget for 3× less wall-clock**. Eight nodes is reachable and is the last point where `grad_accum` can hold the global batch at 32, but 4 nodes is measured and queues faster.

GPU utilization was **86–99%** at `grad_accum ≥ 2`. **Keep `per_device: 1`.**

⚠️ **All of these are self-distillation numbers** — teacher architecture on both sides. The KDA student is ~12% slower at the model level (R0). The 3.01× scaling factor has **not** been re-measured on this model; R1 is the first test of it.

---

## Step 7 — Build the KDA student ✅ **DONE**

> **Full record: `claude/building_the_student.md`.** Only the outcome is here.

Arcee published a converted KDA model but **not** the script that converted it. So their two files were read to learn the *architecture*, and the student was then written from scratch as **subclasses of the current library's Arcee implementation** — which keeps `create_causal_mask`, `GradientCheckpointingLayer` and the transformers 5.x config machinery instead of reimplementing them against 4.57, and reduces the code we own from ~500 lines to ~200.

Every stage was checked against something with a known answer. The student has three config flags that all default to AFM's own behaviour, so with the flags off it must *be* the teacher — and it is, bit for bit. That single property became a regression test that caught two initialization bugs and later proved two packing fixes were inert when they should be.

| | |
|---|---|
| All-attention student vs teacher | **0.000e+00** on logits and all 37 hidden states |
| Hybrid parameter count | **4,999,026,076** — exactly the paper prediction |
| Cost of the architecture changes | NoPE **+3.68 CE**, attention gate **+0.58 CE** |
| Weight transfer into KDA | **−0.21 CE**; hybrid starts at **CE 7.94** |
| KDA vs attention throughput | 0.31× at 2k, crossover ~16k, 1.71× at 32k (single layer) |
| Document isolation, packed sequences | **0.000e+00** in both layer types, under probe |

⭐ **And then R0 confirmed it end to end.** A5b's offline **7.9412** met DistillKit's in-pipeline **7.918** — two independent paths, three significant figures, validating build, save, reload, tokenizer and loss wiring in one number.

⚠️ **The document-isolation machinery is inert in practice** — the pipeline never delivers a segment mask, and the teacher could not use one anyway. Settled in `claude/first_distillkit_runs.md` Part 4.

**Built and committed:** `afm_kda` at `3aafffc`; four students in `$WORK_LUKAS/students/`.

---

## Step 8 — Training infrastructure ✅ **closed; R1 submitted**

> From here the live document is **`claude/first_distillkit_runs.md`**.

### 8a. Token budget → `max_steps` ✅ **arithmetic settled, target open**

```
tokens per step = per_device × grad_accum × total_GPUs × sequence_length
                = 1 × 2 × 16 × 2048 = 65,536          (at 4 nodes)
max_steps       = target_tokens ÷ 65,536
```

**R0 measured 7,585 tok/s at 1 node**; × 3.01 ≈ **22,800 tok/s at 4 nodes**.

| Target | max_steps | wall-clock @ 4 nodes | local hours | share of one epoch |
|---|---|---|---|---|
| **300 M** *(Arcee's stage-1 scale)* | 4,580 | **~3.7 h** | ~470 | 13% |
| **1.3 B** | 19,840 | **~15.8 h** | ~2,025 | 58% |

**300 M is a principled target** — Arcee report stage-1 convergence at token-averaged KL ≈ 0.2 there, so it is the comparable point. **1.3 B is a budget-shaped number** with no convergence evidence behind it.

✅ **R1 launched at 300 M.** The right way to settle the rest is to look at the held-out KL curve and extend from a checkpoint if it is still falling — which turns the question for Anamaria from "guess a budget" into "here is the curve, do we buy more?"

⚠️ **Use `max_steps`, not `num_train_epochs`.** Epochs depend on `num_samples`, so two runs with different data volumes stop being comparable. A fixed token budget is the invariant.

### 8b. Pre-shuffling the dataset ⬜ **downgraded to optional**

The original idea: pre-shuffle the train split on disk so `seed: null` could skip DistillKit's own shuffle and read sequentially, avoiding random seeks into a 20 GB memory-mapped file.

**That was aimed at `save_to_disk`'s cost, and deleting `prepared_dataset_path` removed the cost instead.** The remaining shuffle is a lazy index permutation feeding a `num_proc: 16` tokenize pass — minutes, not hours.

⚠️ **`seed: 42` is REQUIRED, not optional.** The train set is four parquet shards concatenated in file order, so `seed: null` today means `num_samples` takes the first N documents — essentially one crawl segment. **Always shuffle.**

### 8c. Checkpoint resume ⬜ **downgraded; required before any long run**

The test: 20 steps with `save_strategy: steps`, `save_steps: 10`; confirm `checkpoint-10` appears; relaunch with `resume_from_checkpoint: <path>` and confirm the log resumes at step 11. Twenty minutes.

**For a ~4 h run, skipped deliberately.** R1 has `save_steps: 500` and `save_total_limit: 1` enabled, so a failure costs at most 500 steps and resume can be debugged then. **Do the test properly before any 1.3 B run**, where a failure at hour 14 is a different kind of loss.

### 8d. Teacher baseline ⬜ **does not block anything**

Every number in the thesis is a *ratio* to the teacher's. Run AFM-4.5B-Base through the lm-eval harness with the settings you'll use for students, then never change those settings.

⚠️ Two eval capabilities are not built yet: **batched left-padded generation** needs the 2-D padding mask routed through to KDA, and **generation of any kind** needs a working recurrent cache (`use_cache=False` today). MMLU, HellaSwag, ARC and WinoGrande need neither; **IFEval and GSM8K need both.**

---

## Checklist

```
--- STARTUP PHASE: COMPLETE ---
[x] 0. exports appended to ~/.bashrc; $HOME cleaned
[x] 1. FineWeb-Edu downloaded; data_manifest.json saved
[x] 2. fla 0.5.2 installed; KDA verified on A100 at both 2048 and 2560
[x] 3. Arrow dataset + held-out split; 3.015 B GPT-2 tokens, ~2.26 B effective
[x] 4. deepspeed 0.19.6 installed; ds_config_zero2.json written
[x] 5. AFM cache verified; architecture recorded; both smoke tests passed
[x] 5. layer mapping settled -- `all` is correct
[x] 6. multi-node fixed and verified; memory model confirmed to 0.03%
[x] 6. node count chosen: 4 nodes
[x] 7. student built, verified and committed -- claude/building_the_student.md
[x] 8a. throughput MEASURED (7,585 @ 1 node); budget table rebuilt on it
[x] 8. R1 submitted at 300 M tokens, 2026-09-18

--- OPEN, NOT BLOCKING ---
[ ] 8a. final token budget with Anamaria, decided from R1's KL curve
[ ] 8b. pre-shuffle -- optional; seed: 42 is mandatory meanwhile
[ ] 8c. checkpoint resume -- required before any 1.3 B run
[ ] 8d. teacher baseline evals, protocol frozen
[ ] eval: batched left-padded generation needs the 2-D mask routed to KDA
[ ] eval: IFEval and GSM8K need a working recurrent cache
```

**Everything after this lives in `claude/first_distillkit_runs.md`.**

---

# PART 2 — Things that are easy to overlook

## On Leonardo

**Set `save_total_limit`** — and know it does not stop the *final* save. A full-parameter DeepSpeed checkpoint was 60–80 GB; the frozen student's is ~21 GB (10 GB bf16 weights + ~11 GB Adam state over 945 M trainable).

**Use `boost_qos_lprod` for real runs** — 4-day walltime, up to 8 nodes / 32 GPUs. **`boost_qos_dbg` for anything under 30 minutes** — priority 80 vs 40, **max 2 concurrent**, hard 00:30:00 cap. **`lrd_all_serial` for CPU-only work** — budget-free, 4 h, 4 cores, 15400 MB that cannot be raised.

**Add `#SBATCH -A EUHPC_D34_049`.** It works without it now because you have one account; it silently bills the wrong one if that changes.

**A variable assigned inside an sbatch script beats `--export`.** Use `${VAR:-default}` if the submission should win.

**Ask for a walltime you'll use.** A 20-hour request is harder to backfill than an 8-hour one, so it can queue longer.

**Set JIT cache directories per job.** `TRITON_CACHE_DIR` and `TORCH_EXTENSIONS_DIR` default to shared-filesystem paths where concurrent ranks race.

**Compute nodes have no internet.** `HF_HUB_OFFLINE=1`, `WANDB_MODE=offline`. Every *new* model, tokenizer, dataset or remote-code file needs a login-node download first — and **"the cache directory exists" is not proof**.

**A SLURM batch script reads neither `~/.bashrc` nor `~/.bash_profile`.**

**Prefix every one-off model load with `HF_HOME=$WORK_LUKAS/hf_cache`.** A bare `python -c "...from_pretrained..."` re-downloaded 9 GB into `$HOME`.

**`OMP_NUM_THREADS` / `MKL_NUM_THREADS` matter.** Four processes × 32 threads on a 32-core node means 128 threads fighting over 32 cores. Set 8 each.

**`Triton is not supported ... roll back to CPU` is normal on a login node and fatal inside a job.**

**Budget ~4 minutes of Triton autotuning on the first step of every KDA job.** Never quote `train_runtime` from a short run.

## Environment and HuggingFace

**Dry-run before every install into `distill_env_cu126`.** `pip install --dry-run <pkg> | tail -30`. If it proposes touching `torch`, `triton`, or any `nvidia-*` package, stop — that environment has a hand-built `flash_attn` compiled against this exact torch.

**Pin and record versions.** `pip freeze` after every environment change, dated. **And record the fla version in every run**, since the `cu_seqlens` behaviour is specific to 0.5.2.

**A library file you downloaded is not the library you installed.** GitHub's default branch is not the release tag, and for `fla/layers/kda.py` they differ in exactly the part that matters.

**transformers 5.x renamed `torch_dtype` to `dtype`**, and `PretrainedConfig` to `PreTrainedConfig`. Configs are now `@strict` dataclasses with no `__init__`.

**For a remote-code model, `config.json` is the contract; the class is decoration.** A class attribute never reaches the saved config — `to_dict()` copies `self.__dict__`.

**HF copies only ONE level of relative imports for remote code.** Make every file in the closure a first-level import of the entry module, and clear `$HF_HOME/modules/transformers_modules/<model>` after any `.py` change or rename.

**AFM's RoPE config is internally inconsistent** — explicit YaRN factor 20.0 vs an implicit 16.0. Harmless at 2048, and moot under `use_rope: false`.

**AFM uses ReLU² instead of SwiGLU** (so its MLP has no `gate_proj`) and grouped-query attention with 20 query heads over 4 KV heads.

## DistillKit

**⚠️ Read how a flag is consumed before building on what its name implies.** `functionary_packing` reads as "enable packing"; it means "assume the data is already packed in functionary's format", and its entire implementation is two lines in `main.py`. **It does nothing without `dataset.prepacked: true`** — they are two halves of one feature, and enabling only one is silent. This cost the most time of anything in the project.

**⚠️ `lr_scheduler_kwargs: {min_lr: …}` crashes under DeepSpeed.** `cosine_with_min_lr` divides by `optimizer.defaults["lr"]`, which a `DeepSpeedZeroOptimizer` does not have. Use **`min_lr_rate`** (the fraction of peak) — and remember it is then coupled to `learning_rate` by hand.

**⚠️ `prepared_dataset_path` is a trap at scale.** `save_to_disk` runs at ~60 examples/s; helpful at 20k documents, ~2 hours at 500k — *inside the training job*. Leave it out for real runs. Its key is `sha256({path, seed, num_samples})`, so it also silently **hits** if a dataset's contents change at the same path. Version dataset directories by name.

**Datasets take no `kind:` field.** `DatasetPath = HfRepoDataset | LocalDataset` is a *plain* union — pydantic picks by field name. The **teacher** union *is* discriminated, so `kind: hf` there is required.

**Default log level is WARNING.** Pass `-v` (INFO) or `-vv` (DEBUG).

**The tokenizer comes from the *teacher*** when the teacher is `kind: hf`.

**`per_device_eval_batch_size` defaults to 8.** Set it to 1, and set `prediction_loss_only: true`.

**`train_runtime` includes evaluation time** — and, on a short run, the Triton autotune.

**`save_model` runs unconditionally at the end of `do_distill`**, always to `output_path`. A `distillkit_config.yaml` snapshot lands there at startup too — the authoritative record of what a run used.

**Loss weights are normalized automatically — only the *ratio* matters.**

**`num_train_samples` is arithmetic, not a count of what was seen.** Pair it with a step count.

**Rank-0 vs gathered metrics.** `loss` is averaged across ranks; `distillation_loss/*` comes from rank 0 and prints once per **micro-batch**, not per optimizer step. **Under multi-node, `train_runtime` should appear exactly once**, while per-rank `print`s appear once per process — `Froze 147 tensors` four times at one node is a consistency check, not a bug.

**`hs_cosine` costs ~20% of step time.**

**Verify `frozen_res` froze what you meant.** `main.py` prints `Froze N tensors by regular expression` — **147** for our hybrid. And it matches with `re.search`, so anchor every pattern.

**Third-party code may call `AutoConfig.from_pretrained` without `trust_remote_code`.** A remote-code student breaks it.

**`layer_mapping: all` is correct — it aligns layer *inputs*.** See 5f.

**Stage 1 needs an online teacher.** ~9.2 GB per GPU, and it does *not* shrink with more nodes.

**DistillKit is a research release, and yours is a fork.** When behaviour confuses you, read the source.

## Training

**Stage-1 learning rate is 1e-3, and only attention layers train.** Your practical-work instinct says 1e-4; that is the most likely way to conclude "stage 1 doesn't work" when it does.

**Step-0 cross-entropy should be ≈ 7.94, not ≈ 2.2.** The second means the teacher loaded where the student should have.

**`warmup_ratio` × a small `max_steps` is no warmup at all.** 0.025 × 20 steps = half a step. A short run's loss spike can be entirely an artifact of its length.

**Log gradient norms and watch the first 100 steps.** `max_grad_norm: 0.5` clips actively — linear-attention training spikes.

**Held-out KL is your fast metric.** Arcee converged to ~0.2 — but on a different corpus, so treat it as an order of magnitude.

**Fix the global batch at 32 and hold it for every run in the thesis.**

## Evaluation

**Evaluate the teacher first, and freeze the protocol.** **AFM-4.5B-Base is a base model — no chat template.** **Record standard errors.** **Measure throughput, memory and KV-cache size** — the entire justification for the project.

## Bookkeeping

**Git from day one**, in both repositories. Commit before every run; put the short hash in the run name. A run launched from a dirty tree is reproducible only through the `distillkit_config.yaml` snapshot in its output directory.

**Keep a run log**: date, job ID, git commit, config path, **node count**, **grad_accum**, token budget, final loss, held-out KL, peak memory, eval results, one line on what you were testing.

**Back up off Leonardo.** Results, logs, plots, manifest — not checkpoints.

**Keep the process slides current.** Reconstructing a week of work from memory is far more expensive than writing three bullets a day.

**Start writing in week three.** **Hard experiment freeze at end of week seven.** **Get the institute's thesis template early.**

---

# PART 3 — Multi-node training

*Line-by-line explanation in `claude/sbatch_explained.md`; debugging record in `claude/multinode_debug_notes.md`.*

## The working launch script

```bash
#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J distill_kda
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4                   # <-- the only line you change to scale
#SBATCH --ntasks-per-node=1         # ONE task per node. accelerate spawns the GPUs.
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.err

source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate

export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS

# --- rendezvous: every node must agree on one address and port ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export TOTAL_PROCS=$((SLURM_NNODES * GPUS_PER_NODE))   # NOT "WORLD_SIZE"

export DS_CONFIG=$WORK_LUKAS/distillkit/ds_config_zero2.json
export RUN_CONFIG=stage1.yaml

export HF_HOME=$WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=$WORK_LUKAS/wandb
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

cd $WORK_LUKAS/distillkit
echo "master=$MASTER_ADDR nodes=$SLURM_NNODES procs=$TOTAL_PROCS"

srun bash -c '
  export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_PROCID}
  export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}_${SLURM_PROCID}
  accelerate launch \
    --use_deepspeed \
    --deepspeed_multinode_launcher standard \
    --deepspeed_config_file $DS_CONFIG \
    --num_machines $SLURM_NNODES \
    --num_processes $TOTAL_PROCS \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    $(which distillkit) -v $RUN_CONFIG
'
```

**Three things that are not optional.** `--deepspeed_multinode_launcher standard`, or each node builds its own world. `--ntasks-per-node=1`, or SLURM starts four tasks that each start four processes. And **single quotes** around the `srun bash -c` body, so `$SLURM_PROCID` expands per node — with double quotes every node claims rank 0 and the job hangs at rendezvous with no error.

**Do not name a variable `WORLD_SIZE`** — torch.distributed reads that from the environment itself.

⚠️ **A remote-code student needs a single-process pre-warm before `srun`**, because with `--ntasks-per-node=1` all four local processes share one `SLURM_PROCID` and a per-rank module cache does not separate them.

## Verify every multi-node run before trusting it

```bash
J=<jobid>; F=logs/<name>-$J.out
grep -c "train_runtime" $F                          # MUST be 1
grep -o "nranks [0-9]*" $F | sort -u                # MUST match total GPUs
grep -o "nNodes [0-9]*"  $F | sort -u               # MUST match node count
grep -o "train_mem_gpu_alloc_delta': [0-9]*" $F     # must move with node count
grep -c "eval_loss" $F                              # MUST be 0 for a timing run
grep -c "'loss':" $F                                # = optimizer steps (checks grad_accum)
grep -c "roll back to CPU" $F                       # MUST be 0
```

`alloc_delta` is the physical invariant. **If it doesn't move with node count, the job didn't merge.** It is *not* affected by `grad_accum`. ⚠️ **Refit the constants for the frozen student** rather than reusing the teacher's.

The middle three all caught real errors: eval contamination in a timing run, a `grad_accum` that hadn't changed, and a four-way job split.

## ⚠️ The thing that will silently confound your results

```
global_batch = per_device_batch × grad_accum × total_gpus
```

**Rule: fix the global batch once — 32 sequences — and hold it for every run in the thesis:**

| Nodes | GPUs | per_device | grad_accum | global batch |
|---|---|---|---|---|
| 1 | 4 | 1 | 8 | 32 |
| 2 | 8 | 1 | 4 | 32 |
| 4 | 16 | 1 | 2 | 32 |
| 8 | 32 | 1 | 1 | 32 |

Eight nodes is the floor. **Keep one YAML per node count.**

## Other multi-node specifics

**Checkpointing gets slower and more important.** Verify resume (8c) before trusting a long 4-node job.

**Keep `per_device: 1`.** Utilization is already 86–99% at `grad_accum ≥ 2`, and the student's packing helper assumes batch 1.

**Only rank 0 should write logs.** HF `Trainer` handles this for wandb and stdout, but any `print` you add fires once per process.

**Queue time grows with node count** — and with requested walltime.

---

# PART 4 — Leonardo facts (from the CINECA docs, verified 2026-09-08)

## Booster node hardware

- 4 × NVIDIA A100 custom, **64 GiB HBM2e**, NVLink 3.0 (200 GB/s intra-node)
- 1 × Intel Xeon Platinum 8358, **32 cores**
- **512 GiB** DDR4 RAM
- Mellanox HDR InfiniBand, DragonFly+ topology — 4 HCAs per node
- 3,456 nodes total

## Partitions and QoS

| Partition | QoS | Max walltime | Max resources | Priority | Notes |
|---|---|---|---|---|---|
| `lrd_all_serial` | normal | 04:00:00 | 4 cores, **15400 MB RAM** | 40 | **Budget-free**, no GPUs |
| `boost_usr_prod` | `normal` | 24:00:00 | 64 nodes | 40 | |
| `boost_usr_prod` | `boost_qos_dbg` | **00:30:00** | 8 nodes / 32 GPUs | 80 | **Max 2 concurrent** — debugging |
| `boost_usr_prod` | `boost_qos_bprod` | 24:00:00 | 65–256 nodes | 60 | Not applicable |
| `boost_usr_prod` | **`boost_qos_lprod`** | **4-00:00:00** | **8 nodes / 32 GPUs** | 40 | **Real runs** |

The `lrd_all_serial` memory cap **cannot be raised** — `--mem=32G` is rejected with `QOSMaxMemoryPerUser`.

## Accounting

**BH = T · N · R · C** — elapsed hours × nodes × largest resource fraction × CPUs per node (32).

- Full node, 4 GPUs → R = 1.0 → **32 local hours per wall hour per node**
- 1 GPU + 8 cores → R = 0.25 → **8 local hours per wall hour**
- 1 GPU + 32 cores → R = 1.0 (CPU dominates) → **32 local hours per wall hour** ← the trap

| Nodes | GPUs | Local hours per wall hour | Wall hours from remaining 108,913 |
|---|---|---|---|
| 1 | 4 | 32 | ~3,400 |
| 2 | 8 | 64 | ~1,700 |
| 4 | 16 | 128 | ~850 |
| 8 | 32 | 256 | ~425 |

**A job that scales perfectly costs the same total budget regardless of node count.** At the measured efficiencies, 2 nodes costs ~13% more than 1 for 1.8× less wall-clock, and 4 nodes ~33% more for 3.0× less.

**Budget linearization:** monthly quota = total ÷ months. Within quota, full priority; beyond it, priority decays but jobs still run.

## Filesystems

| Area | Quota | Backup | Cleaning | Retention |
|---|---|---|---|---|
| `$HOME` | 50 GB | *not active* | none | while username active |
| `$WORK` | 2 TB | none | none | 6 months post-project |
| `$FAST` | 1 TB | none | none | 6 months post-project |
| `$SCRATCH` | no limit | none | **files >40 days deleted** | — |
| `$PUBLIC` | 50 GB | none | none | while username active |
| `$TMPDIR` | **10 GB, RAM-based on Booster** | none | end of job | per job |

**Settled layout:**

| What | Where | Why |
|---|---|---|
| Datasets, HF cache, checkpoints, output, code | `$WORK` | Documented home for batch output. ~1.3 TB free (verified 2026-09-18). |
| Python venvs | `$HOME` | 14 GB, not shared, never purged, reproducible anyway |
| pip / XDG caches | `$WORK` (via `.bashrc`) | keeps `$HOME` from silently refilling |
| Large temporary files during a run | `$SCRATCH` | no quota; 40-day purge |
| Triton + torch-extension caches | `/tmp` (node-local, per job+rank) | tens of MB; must not be on a shared filesystem |
| — | `$FAST` | **Not needed** unless dataloading is *measured* as a bottleneck |

**`$TMPDIR` / `/tmp` on Booster is RAM-backed and fixed at 10 GB.** Fine for JIT caches; never large intermediates.

`$SCRATCH` purging writes its deletion log to `$HOME` as `CLEAN_<yyyymmdd>.log`.

## Sources

- [Leonardo cluster page](https://docs.hpc.cineca.it/hpc/leonardo.html)
- [Introduction / accounting](https://docs.hpc.cineca.it/hpc/hpc_intro.html)
- [File systems and data management](https://docs.hpc.cineca.it/hpc/hpc_data_storage.html)
- [Scheduler and job submission](https://docs.hpc.cineca.it/hpc/hpc_scheduler.html)
- CINECA user support: superc@cineca.it