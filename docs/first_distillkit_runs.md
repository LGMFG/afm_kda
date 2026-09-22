# First DistillKit Runs — Stage 1 with the KDA Student

*The live document. `claude/startup_plan_and_pitfalls.md` = infrastructure (closed). `claude/building_the_student.md` = the model, Tracks A/B/C.*

*Revision 17 (2026-09-20) — **the blog post's token budget is now known, and the benchmark comparison is placed.** Their evaluated students had **300 M stage 1 + ~1 B at 32k with MLPs unfrozen**; ours is stage 1 only. Matching their teacher row shows their chart uses ARC-C **plain acc** and HellaSwag **acc_norm** — raw, that is **47.78 vs their 48.5** on ARC-C and **72.40 vs their 74.3** on HellaSwag, with the same teacher on both sides — on 23 % of the tokens. ⚠️ Relative-score figures quoted anywhere in this document are **computed here from those raw numbers**, not published by Arcee. ⭐ And their own numbers show **KDA-only beating the hybrid on MMLU and ARC-C**, which together with B9a's 95.7× memory figure makes `kda_only` the most valuable second run.*

*Revision 16 (2026-09-20) — **held-out CE measured: the student closed 97.2 % of the initialization gap.** 2.4559 against the teacher's 2.2963 on 131 k held-out tokens, perplexity 11.7 vs 9.9. A5b's 2.3030 and 7.9412 both reproduce, so the axis is sound. **Identical numbers under transformers 5.8.1 and 5.9.0** — the env-skew question is closed by measurement. Also: where the 300 M token count is confirmed in the log, and why `save_total_limit: 1` is the wrong setting for the next run.*

*Revision 15 (2026-09-20) — **the R1 student is benchmarked.** ARC-C 0.5205, HellaSwag 0.7240, WinoGrande 0.6630 against a teacher at 0.5922 / 0.7816 / 0.7048 — **79%, 89% and 80% of the teacher's above-chance score, after stage 1 alone.** MMLU pending on both sides. Also settles the transformers-version question: **do not downgrade to match the reference**, and check the env skew by measurement rather than by argument.*

*Revision 14 (2026-09-20) — **corrects revision 13's account of the KL gap.** I gave three reasons why our KL ≈ 0.35 differs from Arcee's ≈ 0.2; reading `blogpost_yaml.txt` line by line kills two of them. Same teacher, same loss weights, **same KL temperature**, same LR, same schedule shape and floor. **One clear difference remains: the corpus.** Also: `save_pretrained` *does* copy the remote-code files into the checkpoint — the real trap is the module cache path.*

*Revision 13 (2026-09-19) — **R1 finished, and it worked.** 4,580 steps, 300 M tokens, 3.27 h, no crash. **hs_cosine 0.841 → 0.070, KL 10.13 → 0.35, CE 7.918 → 2.26**, held-out loss monotonically down at all nine eval points. Also: the **teacher baseline** for three of the four benchmarks, and a new trap — the sbatch's provenance line records the *repo* hash, not the code the model actually ran.*

*Revision 12 — restructured; A9 closed document isolation in favour of path (a).*

---

## 🧭 How to read this document

| Part | What's in it | When you need it |
|---|---|---|
| **1. Now** | status, what R1 produced, next action, open decisions | every day |
| **2. The runs** | run log, results, what R0 and R1 each established | after each job |
| **3. Reference** | config settings and why, pre-flight, budget arithmetic | when writing a config |
| **4. Closed** | document isolation — the whole story, settled | when writing the thesis |
| **5. Traps** | the accumulated list | before every run |

---

# PART 1 — NOW

## Status

| | |
|---|---|
| Students | ✅ four, built from `3aafffc` |
| Pipeline | ✅ R0 clean, **R1 clean** |
| Document isolation | 🔵 closed — path (a), see Part 4 |
| **R1 — stage 1, 300 M tokens** | ✅ **COMPLETE, job 58141472** |
| Teacher baseline | ✅ ARC-C / HellaSwag / WinoGrande · ⬜ MMLU |
| **Student benchmarks** | ✅ **ARC-C / HellaSwag / WinoGrande** · ⬜ MMLU · ⬜ held-out CE |
| Generation (GSM8K, IFEval) | 🔶 Track B — `building_the_student.md` |
| GDN sibling student | ⬜ Track C |

## ⭐ R1 worked

| Signal | At init | After 300 M tokens | Target |
|---|---|---|---|
| **`hs_cosine`** ⭐ | 0.841 | **0.070** | *falling* — the pre-registered pass condition |
| **KL vs teacher** | 10.13 | **~0.35** | Arcee report ≈ 0.2 at this scale |
| **Cross-entropy** | 7.918 | **~2.26** | teacher's own held-out CE is 2.3030 |
| Held-out combined loss | — | **0.5686** | monotone down at all 9 eval points |
| `grad_norm` | 6.36 @ step 10 | 0.057 | no spike, no NaN |

⭐ **`hs_cosine` was the number the doc pre-committed to before the run**, precisely because R0's 20 steps had shown it *rising*. It fell by 92%. Writing the pass condition down in advance is what makes this a result rather than a post-hoc reading.

⭐ **CE went from 7.918 to ~2.26, and the teacher's held-out CE is 2.3030.** Different data, so not a like-for-like comparison — but the gap the student had to close was 5.6 nats and essentially all of it is gone. A hybrid whose token-mixing layers were random noise nine hours ago is now a language model.

### ⭐ On KL ≈ 0.35 against Arcee's ≈ 0.2 — what is actually different

⚠️ **Revision 13 offered three reasons and two of them were wrong.** Line-by-line against `blogpost_yaml.txt`:

| | Arcee | Ours | Same? |
|---|---|---|---|
| teacher | `arcee-ai/AFM-4.5B-Base` | same | ✅ **identical pairing** |
| loss weights | 0.2 / 0.2 / 0.6 | same | ✅ |
| **KL temperature** | **2.0** | **2.0** | ✅ **the same quantity is being compared** |
| `layer_mapping` | `all` | same | ✅ |
| `learning_rate` / `warmup_ratio` | 1e-3 / 0.025 | same | ✅ |
| scheduler | `cosine_with_min_lr`, floor **1e-5** | same floor, via `min_lr_rate` | ✅ |
| `max_grad_norm`, `weight_decay`, `optim`, `bf16` | 0.5 / 0.00 / adamw_torch / true | same | ✅ |
| `sequence_length`, `packing`, `pad_to_multiple_of` | 2048 / True / 128 | same | ✅ |
| **dataset** | `arcee-train/afm-autodistill-mix-v0` — internal **instruction mix** | **FineWeb-Edu** — raw web text | ❌ |
| schedule length | `num_train_epochs: 1` over an **unknown-size** corpus | `max_steps: 4580` = exactly 300 M | ❌ |
| global batch | 4 × 2 × *N* GPUs, *N* unpublished | 1 × 2 × 16 = **32** | ❓ |
| `frozen_res` norms | bare `norm\.weight` — also freezes KDA's `o_norm` | anchored; **we train `o_norm`** | ❌ minor |

**So the "different teacher–student pairing" reason was simply false**, and "a schedule that hit its floor" is not a difference either — theirs anneals to the same 1e-5. What survives:

1. ⭐ **The corpus.** KL measures agreement with the teacher *on the data you show it*. FineWeb-Edu is raw web text; their mix is curated instruction data. A teacher's next-token distribution on curated instruction text is plausibly easier to match than on arbitrary web prose. **This is a mechanism, not a measurement** — the honest form for the thesis.
2. **Schedule length is not comparable.** `num_train_epochs: 1` stretches their cosine over one epoch of a corpus whose size they never state. Their "KL ≈ 0.2 at ~300 M tokens" may therefore be a point *passed through* mid-schedule rather than an endpoint. Ours is an endpoint by construction.
3. **Their global batch is unknown** and, at `per_device 4`, likely larger than our 32.

**How to write it:** ours reached token-averaged KL ≈ 0.35 at 300 M tokens at temperature 2.0, against ≈ 0.2 reported by Arcee at the same token count, under an otherwise identical recipe, on a different and less curated corpus. State the identity of the hyperparameters — that is what makes the one remaining difference interpretable.

### ⭐ Held-out cross-entropy — the single cleanest result

`test_r1_ce.py`, 64 × 2048 = 131,072 held-out tokens, sdpa, bf16:

| model | CE (first 8) | CE (all 64) | ± se | perplexity | vs teacher |
|---|---|---|---|---|---|
| **Teacher** AFM-4.5B-Base | **2.3030** | 2.2963 | 0.0400 | **9.9** | — |
| hybrid31, **untrained** | 7.9274 | 7.9398 | 0.0195 | 2,806.8 | +5.6435 |
| **hybrid31, R1 @ 300 M** | 2.4383 | **2.4559** | 0.0392 | **11.7** | **+0.1595** |

⭐ **The student closed 5.484 of the 5.644-nat gap — 97.2 % — in 300 M tokens.** Perplexity 11.7 against the teacher's 9.9, i.e. **17.3 % higher** (`exp(0.1595) = 1.173`). That single sentence is the most compact statement of what stage 1 achieved.

**The first-8 column is the validity check, and it passes.** The teacher reads **2.3030**, reproducing A5b's published number exactly — the script is on the same axis. The untrained hybrid reads 7.9274 against A5b's 7.9412 and R0's 7.918: three measurements of "the untrained hybrid" spanning 0.023 nats.

📌 **That small spread is itself informative.** A5b built the model in memory; this loads the saved checkpoint, which was built by `build_student.py` with its own ordering of random draws. **The ★ parameters are random, so "the untrained hybrid" is a distribution, not a point**, and ~0.02 nats is the scale of that seed noise. Any claimed effect smaller than that is noise.

⚠️ **The printed standard errors are the wrong error bar for the comparison.** ±0.040 is the spread *across sequences*, which is dominated by how hard each passage is — and both models see the same passages. The right statistic is the **paired** per-sequence difference, whose standard error is far smaller. The gap survives either way (0.16 is 4× the unpaired se), but quote the paired figure in the thesis.

⭐ **Cross-environment check: identical numbers under transformers 5.8.1 and 5.9.0.** This is what settles the version question — by measurement rather than argument. **The 5.8.x/5.9.x skew does not move this model's logits.** Record both versions with the numbers and stop worrying about it.

**How the CE gap and the benchmark gap fit together.** A 17 % perplexity penalty coexists with 79–89 % relative retention on discriminative benchmarks. That is the expected shape: a modest language-modelling penalty shows up amplified when a task forces a choice between close alternatives.

### ⭐⭐ Can we compare to the blog post? Yes — and the comparison is flattering

**Their training budget, from the post's own words:**

> "…only about 300 million tokens seen, token-averaged KL divergence between student and teacher converged to around 0.2"
>
> "one-phase distillation, plus about a billion tokens at 32k sequence length, with a 1e-5 learning rate"
>
> "barely 1 billion for the distilled students"

So their **evaluated** models are **300 M (stage 1, 2048) + ~1 B (32k, `lr 1e-5`, MLPs unfrozen)** ≈ **1.3 B tokens**. Ours is **300 M of stage 1 only, MLPs frozen throughout** — about **23 %** of their budget, and missing the stage in which their MLPs were allowed to adapt.

⭐ **Which metric they report — established by matching the teacher row.** Their chart's teacher reads ARC-Challenge **55.6** and HellaSwag (Norm) **78.0**. We measured our teacher at ARC-C `acc` **55.63** and HellaSwag `acc_norm` **78.16**. So their "ARC-Challenge" is **plain `acc`**, their HellaSwag is **`acc_norm`**, and it is the same teacher. **Compare on those metrics, not on ARC-C `acc_norm`.**

#### The raw numbers — what is actually published and what we actually measured

| | teacher | hybrid student | source |
|---|---|---|---|
| **ARC-C `acc`** — theirs | 55.6 | **48.5** | their chart, as printed |
| **ARC-C `acc`** — ours | 55.63 | **47.78** | job 58216691 / our student run |
| **HellaSwag `acc_norm`** — theirs | 78.0 | **74.3** | their chart, as printed |
| **HellaSwag `acc_norm`** — ours | 78.16 | **72.40** | ditto |
| MMLU `acc` 5-shot — theirs | 63.1 | 55.1 | their chart |
| MMLU `acc` 5-shot — ours | ⬜ | ⬜ | queued |

**Read across those rows and the comparison is direct: 47.78 against 48.5, and 72.40 against 74.3, with the same teacher on both sides.**

#### The derived numbers — ⚠️ computed here, not published anywhere

⚠️ **Nothing below appears in their chart.** These are the raw numbers above put through RADLADS's relative-score formula, which is worth doing because it is the metric the paper we are replicating uses and because a raw point gap flatters a task with a high chance floor:

```
relative score = (student − chance) / (teacher − chance)      chance: 0.25 ARC-C, 0.25 HellaSwag
theirs, ARC-C     = (48.5  − 25) / (55.6  − 25) = 23.5 / 30.6 = 76.8 %
ours,   ARC-C     = (47.78 − 25) / (55.63 − 25) = 22.78 / 30.63 = 74.4 %
theirs, HellaSwag = (74.3  − 25) / (78.0  − 25) = 49.3 / 53.0 = 93.0 %
ours,   HellaSwag = (72.40 − 25) / (78.16 − 25) = 47.40 / 53.16 = 89.2 %
```

| relative score *(computed)* | **ours** (300 M, stage 1 only) | **theirs** (≈1.3 B, + 32k stage) |
|---|---|---|
| ARC-Challenge | **74.4 %** | 76.8 % |
| HellaSwag | **89.2 %** | 93.0 % |
| MMLU | ⬜ | 79.0 % |
| GSM8K | 🔶 Track B | 70.1 % |

⭐ **Within 2–4 points of their published hybrid, on 23 % of the tokens and without stage 3.** The plausible mechanism is that their extra ~1 B tokens went into **32k long-context** capability, which none of these 2048-length benchmarks tests — so the extra budget bought something the benchmarks cannot see. That is a claim about where the training went, and it is worth stating as such rather than as a victory.

**So the two comparisons have different standing, and the thesis must say which is which:**

| comparison | like-for-like? |
|---|---|
| **KL ≈ 0.35 vs their ≈ 0.2 at 300 M** | ✅ **yes** — same stage, same token count, same temperature, same recipe. Only the corpus differs. |
| **Benchmarks vs their published chart** | ⚠️ **no** — theirs had 4.3× the tokens and a long-context stage with MLPs unfrozen. Ours is the *stage-1-only* point on their curve, which they never published. |

📌 **That second row is an opportunity, not a weakness.** "What does stage 1 alone buy?" is a question the reference leaves open, and our numbers answer it.

### ⭐ Their own numbers argue for training KDA-only

Raw, as printed in their chart, then the same relative-score transformation ⚠️ *(computed here)*:

| | MMLU | ARC-C | HellaSwag | GSM8K |
|---|---|---|---|---|
| teacher *(published)* | 63.1 | 55.6 | 78.0 | 52.1 |
| hybrid 3:1 *(published)* | 55.1 | 48.5 | 74.3 | 36.5 |
| kda_only *(published)* | 55.8 | 49.9 | 74.3 | 26.8 |
| FLP *(published)* | 44.0 | 42.4 | 62.6 | 21.0 |
| — hybrid, relative *(computed)* | 79.0 % | 76.8 % | 93.0 % | **70.1 %** |
| — **kda_only, relative** *(computed)* | **80.8 %** | **81.4 %** | 93.0 % | 51.4 % |
| — FLP, relative *(computed)* | 49.9 % | 56.9 % | 70.9 % | 40.3 % |

**KDA-only beats the hybrid on MMLU and ARC-C in Arcee's own published numbers** — 55.8 against 55.1 and 49.9 against 48.5 — ties on HellaSwag, and loses only on GSM8K, the one task needing multi-step retrieval over a long chain. Put that beside B9a's memory measurement, where **kda_only is 95.7× smaller at 32k against the hybrid's 3.9×**, and KDA-only is better on *both* axes for everything except math.

⚠️ **That makes `afm_kda_only_init` the most valuable second training run**, ahead of the de-novo and FLP ablations. It is also why `kda_only` generation support (no attention layers → nothing to read a cache length from) stops being a footnote — see `claude/track_b.md`.

### ⭐ The student benchmarks — job of 2026-09-20, `checkpoint-4580`

| Benchmark | Teacher | **R1 student** | Gap | **Relative score** |
|---|---|---|---|---|
| ARC-Challenge, `acc_norm` | 0.5922 | **0.5205** | −7.2 pts | **79.1 %** |
| HellaSwag, `acc_norm` | 0.7816 | **0.7240** | −5.8 pts | **89.2 %** |
| WinoGrande, `acc` | 0.7048 | **0.6630** | −4.2 pts | **79.6 %** |
| MMLU, `acc` (5-shot) | ⬜ | ⬜ | | |

**Relative score** is RADLADS's own metric, `(student − chance) / (teacher − chance)`, with chance 0.25 for the four-way tasks and 0.50 for WinoGrande. It is the right framing because it is the one the paper we are replicating uses, and because raw point gaps flatter tasks with a high floor.

⭐ **What this is.** Twenty-seven of thirty-six token-mixing layers replaced by a recurrent linear-attention mechanism, initialized from noise plus a projection transplant, trained for **300 M tokens of stage 1 alone** — no stage 2, no long-context stage, MLPs and embeddings frozen the whole way — and the model retains **79–89 %** of its teacher's above-chance ability on three standard benchmarks. That is the headline result of the thesis so far.

⚠️ **Do not celebrate before MMLU.** In every published attention-conversion table — RADLADS's own, LoLCATs', the Mamba2-Llama3 rows — **MMLU falls hardest and furthest**. It is the benchmark that most directly probes retrieval of specific facts from a long context, which is exactly what a fixed-size recurrent state is worst at. A student at 79–89 % here could still be at 55–65 % relative on MMLU, and that would be the honest, interesting result rather than a failure.

📌 **The three numbers are not equally informative.** HellaSwag at 89 % is the easiest of the three for a linear-attention model — short contexts, completion plausibility, little long-range retrieval. WinoGrande's 79.6 % rests on a 0.50 chance floor, so its relative score is the most sensitive to noise (±0.013 raw becomes ±6 points relative). ARC-C at 79 % is the most demanding of the three.

⚠️ **The evaluation ran against `output/r1_stage1_hybrid31/checkpoint-4580`, not the top-level directory** — visible from the one `[AFMKDA-PROBE]` block in the log, which comes from that subdirectory's own older `.py` copies. The weights are the same model; only the print statements differ. **Pick one of the two paths and use it for every student number**, or the provenance line in the thesis will not be checkable.

### The run itself

| | |
|---|---|
| Wall-clock | **11,790 s = 3.27 h** (estimated 3.7) |
| Throughput @ 4 nodes | **25,455 tok/s** (estimated 22,800) |
| Implied scaling vs 1 node | **3.36×** — the teacher-measured 3.01× was *conservative* |
| `train_mem_gpu_alloc_delta` | 10.73 GB (R0 at 1 node: 12.86) |
| `epoch` | **0.7655** — predicted 0.765 |
| `Froze 147 tensors` | 16×, one per rank ✅ |
| Triton CPU fallback | 0 ✅ |
| NaN / OOM / traceback | none ✅ |

**The LR schedule did exactly what the config said.** Peak 1e-3 reached after ~114 warmup steps, cosine decay, floor **1e-5** — which is `min_lr_rate: 0.01 × 1e-3`, confirming the DeepSpeed workaround end to end.

**An arithmetic check worth doing once.** The scalar `loss` should be the configured weighted sum of the three components. At the end: `0.2 × 2.2626 + 0.2 × 0.3526 + 0.6 × 0.0705 = 0.5653`, against a reported 0.5621. ✅ **The loss weights are the ones in the YAML.** A silently-ignored weight would have been invisible in any other number.

### The curve, binned by epoch

| epoch | tokens | CE | KL | hs_cosine |
|---|---|---|---|---|
| 0.0 | ~0 | 3.14 | 1.811 | 0.217 |
| 0.1 | 39 M | 2.505 | 0.793 | 0.126 |
| 0.2 | 78 M | 2.427 | 0.606 | 0.105 |
| 0.3 | 118 M | 2.374 | 0.537 | 0.095 |
| 0.4 | 157 M | 2.332 | 0.463 | 0.086 |
| 0.5 | 196 M | 2.308 | 0.398 | 0.079 |
| 0.6 | 235 M | 2.272 | 0.379 | 0.074 |
| 0.7 | 274 M | 2.263 | 0.353 | 0.071 |

*(The 0.0 bin averages over the first 598 micro-batches, which includes warmup — the very first eight read CE 7.5–8.3, KL 8.2–12.6, hs_cosine 0.83–0.85.)*

**Held-out, at each of the nine evals:** 0.7429 → 0.6954 → 0.6549 → 0.6308 → 0.6109 → 0.5923 → 0.5806 → 0.5721 → 0.5688 → 0.5686. **Monotone throughout, no overfitting anywhere.**

⚠️ **Read the flattening at the end correctly.** The last 500 steps moved held-out loss by 0.0002 — but the learning rate had been sitting on its 1e-5 floor for roughly the final 600 steps. **That is the schedule ending, not the model saturating.** The two look identical in the curve and mean opposite things for the budget question.

**The consequence:** extending this run is **not** a `resume_from_checkpoint`. A cosine schedule that has already annealed to its floor cannot be continued — you would be training at 1% of peak LR forever. More tokens means **a new run with `max_steps` set to the larger budget from the start**. Decide the budget before launching, not after.

## ⬅ Next action: evaluate the R1 checkpoint

Three of the four benchmarks need no new code — likelihood scoring works today (Track B covers the rest).

### ✅ The checkpoint is self-contained — verified

⚠️ **Revision 13 warned that `save_model` does not copy the `.py` files. That was wrong.** `save_pretrained` calls transformers' `custom_object_save` for any model loaded with `trust_remote_code`, so the output directory already contained all four architecture files:

```
config.json  configuration_afm_kda.py  configuration_arcee.py
modeling_afm_kda.py  modeling_arcee.py  generation_config.json
tokenizer.json  tokenizer_config.json  model.safetensors (9.998 GB)
checkpoint-4580/  distillkit_config.yaml  training_args.bin  README.md
```

`__init__.py` is absent and is not needed — transformers creates its own package directory under the module cache. Verified load:

```
AFMKDAForCausalLM 4999026076     ✅ exactly the built parameter count
```

⭐ **`distillkit_config.yaml` is saved beside the weights.** That is the run's own record of what it was told to do — worth citing rather than the repo copy, which can drift.

### ⚠️ The real trap: the module cache is under `$HF_HOME`, not `~/.cache/huggingface`

A re-sync of the `.py` files into the student directories was followed by

```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules      # WRONG PATH
```

and the next job still printed the `[AFMKDA-PROBE]` lines from the deleted code, loading from
`$WORK_LUKAS/hf_cache/modules/transformers_modules/afm_kda_hybrid31_init/1581f4f08ffba7bc/`.
**The sbatch sets `HF_HOME=$WORK_LUKAS/hf_cache`, so that is where the module cache lives.** The correct command, and a check that it worked:

```bash
rm -rf $HF_HOME/modules/transformers_modules
ls $HF_HOME/modules/transformers_modules 2>/dev/null || echo "cleared"
```

*A stale module cache is silent: the model loads, runs, and is the old code.*

### Then

1. **Held-out CE**, with `test_a5b.py`'s measurement path pointed at the trained checkpoint. This is the number that sits next to A5b's 7.9412 and the teacher's 2.3030, and it is the cleanest single result in the thesis.
2. **`eval_zeroshot_afm.sbatch`** with `NAME=r1_hybrid31` and `MODEL_PATH=$WORK_LUKAS/output/r1_stage1_hybrid31` — 30-minute debug queue.
3. **`eval_mmlu_afm.sbatch`**, same two edits — long queue.

⚠️ **`eval-env2` now has fla 0.5.2** (installed 2026-09-19; torch, transformers and lm_eval unchanged). Without it the student's remote code raises `ImportError` at load.

## Open decisions — for Anamaria

> **1. Token budget.** 300 M reached KL ≈ 0.35 against Arcee's reported ≈ 0.2, with the held-out curve still falling until the LR floor. 1.3 B is ~12.5 h at the measured 25,455 tok/s and ~2,025 local hours against ~108,000 remaining. Given the curve, is a longer stage 1 the best use of that budget, or are the ablations worth more?
>
> **2. Scope.** Arcee unfreeze the MLPs during a 32k long-context stage we are skipping. A short second run at 2048 with `frozen_res` removed recovers RADLADS's stage 2 for about one extra stage-1-sized run. Worth it?
>
> **3. Ablations.** Four students are built (hybrid 3:1, KDA-only, FLP, no-weight-transfer), each ~3.3 h. How many get a full run?

⭐ **The budget question is now answerable with data rather than a guess**, which is what running 300 M first bought. Benchmark the R1 checkpoint before answering it — a model already close to the teacher on ARC-C and HellaSwag argues for spending the next runs on ablations; a large gap argues for more tokens.

---

# PART 2 — THE RUNS

## Run log

| Run | Date | Job | Nodes | afm_kda / distillkit | Tokens | steps | CE / KL / hs @0 → @end | GB | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R0-a | 09-18 | 58114936 | 1 | 3aafffc / 93031d5 | — | 0 | — | — | **crash**: `min_lr` ÷ DeepSpeed optimizer |
| **R0** | 09-18 | 58117838 | 1 | 3aafffc / 93031d5 | 1.31 M | 20 | 7.92 / 10.13 / 0.841 → 8.46 / 8.58 / 0.968 | 12.86 | 6/7 checks; 7,585 tok/s |
| R0b | 09-18 | 58131180 | 1 | 3aafffc + probe | 0.13 M | 2 | 7.92 / 10.13 / 0.841 | 12.86 | diagnostic: `attention_mask = None` |
| **R1** ⭐ | 09-18/19 | **58141472** | 4 | ⚠️ *repo* 3fb1104 / 2e26723, **model 3aafffc** | **300.1 M** | **4580** | **7.92 / 10.13 / 0.841 → 2.26 / 0.35 / 0.070** | 10.73 | ✅ **3.27 h, 25,455 tok/s, clean** |

*torch 2.12.0+cu126, transformers 5.8.1, fla 0.5.2, deepspeed 0.19.6 throughout.*

⚠️ **The two hashes in R1's row are not the same thing, and this is a trap.** The sbatch's provenance block ran `git log -1` in `afm_kda/` and printed **3fb1104**. But the model that trained is the *checkpoint's* copy of the five `.py` files, frozen at build time — **3aafffc**. Proof from the run's own log: 48 `[AFMKDA-PROBE]` lines appear, and those prints had already been deleted from the repo. **The provenance line records the repo, not the code.** Print `build_info.json`'s hash from the checkpoint instead, or in addition.

## Results

| Metric | Teacher | At init | **After stage 1 (R1)** | After stage 2 (if run) |
|---|---|---|---|---|
| Held-out CE | 2.3030 | 7.9412 (A5b) / 7.918 (R0) | ⬜ *to measure* (train CE ≈ 2.26) | |
| Held-out perplexity | 10.0 | 2,814 | ⬜ | |
| Token-averaged KL | 0 | ~10.1 | **~0.35** | |
| `hs_cosine` | 0 | 0.841 | **0.070** | |
| Held-out combined loss | — | — | **0.5686** | |
| MMLU (5-shot, acc) | ⬜ | — | ⬜ | |
| **ARC-Challenge** (0-shot, acc_norm) | **0.5922** ±0.0144 | — | **0.5205** ±0.0146 | |
| **HellaSwag** (0-shot, acc_norm) | **0.7816** ±0.0041 | — | **0.7240** ±0.0045 | |
| **WinoGrande** (0-shot, acc) | **0.7048** ±0.0128 | — | **0.6630** ±0.0133 | |
| *(ARC-C raw acc)* | 0.5563 | — | 0.4778 | |
| *(HellaSwag raw acc)* | 0.5882 | — | 0.5449 | |
| GSM8K / IFEval | ⬜ Track B | — | ⬜ Track B | |
| Parameters | 4.6192 B | 4.9990 B | 4.9990 B | |
| Throughput @ 4 nodes | — | — | **25,455 tok/s** | |

**Teacher baseline, job 58216691**, 6 minutes on the debug queue, batch size auto-detected at 64. Protocol frozen in `claude/eval_zeroshot_afm.sbatch` and `claude/eval_mmlu_afm.sbatch`: 0-shot for ARC-C, HellaSwag and WinoGrande, 5-shot for MMLU, no chat template — matching RADLADS and the practical work. ✅ **ARC-C and HellaSwag land on the blog post's published teacher figures**, which validates the harness configuration independently.

## What R1 established

Beyond the numbers, five things are now closed that were open before:

1. **The pipeline scales.** 16 ranks, 4 nodes, ZeRO-2, 4,580 steps, no hang, no rank divergence, one `train_runtime`. The multi-node work from the startup phase holds under a real load.
2. **The architecture trains.** 27 randomly-initialized recurrent layers, inside a network frozen everywhere else, learned to reproduce what attention did. This was never guaranteed — it is the project's core hypothesis and it is now evidence.
3. **The freezing worked as designed.** 147 frozen, 477 trainable, on all 16 ranks, and `alloc_delta` fell from R0's 12.86 GB to 10.73 as Adam state sharded sixteen ways.
4. **The estimates were conservative in the right direction.** 3.36× scaling against a predicted 3.01×; 3.27 h against 3.7; `epoch` 0.7655 against a hand-computed 0.765.
5. **R0's alarming `hs_cosine` rise was an artifact of run length**, exactly as the doc argued at the time. 20 steps under a schedule that decays to zero in 20 steps says nothing.

## What R0 established

**Seven checks, six green.**

| # | Check | Measured | |
|---|---|---|---|
| 1 | model class / params | `AFMKDAForCausalLM`, 4999026076 | ✅ |
| 2 | step-0 `cross_entropy` | **7.918** | ✅ |
| 3 | step-0 `kl` | 10.13 | ✅ |
| 4 | step-0 `hs_cosine` | 0.841 | ✅ |
| 5 | `Froze N tensors` | 147, four times | ✅ |
| 6 | no Triton CPU fallback | absent | ✅ |
| 7 | segment mask arrives | **absent** | ❌ → Part 4 |

⭐ **Check 2 validated all of Track A in one number.** A5b measured the untrained hybrid at **7.9412** offline with a hand-written script; R0 measured **7.918** inside DistillKit, on different data, through a saved-and-reloaded checkpoint, with DeepSpeed in the loop. Two independent paths, three significant figures. **≈ 2.2 would have meant the teacher was training itself.**

**Why `Froze 147 tensors` appears once per rank.** It is a bare `print`, not rank-gated. Identical counts are the pass condition; differing counts would mean ranks computing different gradients, silently. Contrast `train_runtime`, which must appear exactly **once**.

**Throughput at 1 node:** `train_runtime` 402.5 s over 20 steps, of which step 1 was 239.3 s of Triton autotuning; steps 2–20 ran at ~8.64 s, giving **7,585 tok/s**. Against the teacher's self-distillation at 10,727, **the KDA student costs ~12% at the model level** — nowhere near A6's single-layer 0.31×, because the MLPs dilute the token-mixing cost.

---

# PART 3 — REFERENCE

## The settings that matter, and why

### The global-batch invariant

```
global_batch = per_device × grad_accum × total_GPUs = 32 sequences, always
```

| Nodes | GPUs | per_device | grad_accum |
|---|---|---|---|
| 1 | 4 | 1 | 8 |
| 4 | 16 | 1 | **2** |

**Fix it once and hold it for every run in the thesis**, or runs stop being comparable. `tokens/step = 32 × 2048 = 65,536` regardless of node count.

⚠️ **`per_device_train_batch_size: 1` is load-bearing twice over** — 4 does not fit in 64 GB, and the model's `build_cu_seqlens` assumes one row.

⚠️ **DeepSpeed's `grad_accum` wins.** The mismatch warning is normal and means the YAML is the single source.

### What stage 1 freezes

| | Frozen | Why |
|---|---|---|
| `embed_tokens`, `lm_head` | ✅ | teacher's — and why `hidden_states[0]` is identically zero in `hs_cosine` |
| `model.norm` + both per-layer layernorms | ✅ | teacher's, unchanged |
| `mlp.up_proj` / `mlp.down_proj`, all 36 layers | ✅ | **where the factual knowledge lives** (RADLADS) |
| everything under `self_attn` | ❌ **trainable** | the only part that is new |

**Where 147 comes from.** 624 parameter tensors total: 1 `embed_tokens` + 1 `lm_head` + 1 `model.norm` + 72 layernorms + 72 MLP = **147 frozen**; 9 attention layers × 5 + 27 KDA layers × 16 = **477 trainable**. ≈ 945 M of 5.0 B, 19%.

**What the norms are.** AFM is pre-norm; each norm is an RMSNorm with a learned per-channel gain. Freezing them is about **measurement integrity**, not capacity. `input_layernorm` is followed by trainable projections so freezing it is **absorbable**; `post_attention_layernorm` and `model.norm` feed frozen matrices, so those are a genuine constraint, accepted on purpose. **A frozen diagonal scale is free only when what follows is trainable and full-rank.**

### ⚠️ `frozen_res` uses `re.search`, not `re.match`

```python
if any(fre.search(name) for fre in frozen_res):
    param.requires_grad = False
```

So the blog post's bare `norm\.weight` also hits KDA's own `o_norm.weight`. Severity is low — `o_norm` starts at ones and a fixed diagonal scale upstream of a trainable full-rank `o_proj` costs no expressivity — but anchor anyway. Ours are anchored, and R0 and R1 both printed the predicted 147/477.

### ⚠️ `min_lr` crashes under DeepSpeed — use `min_lr_rate`

Job 58114936 died on all four ranks before step 0:

```
min_lr_rate = min_lr / optimizer.defaults["lr"]
AttributeError: 'DeepSpeedZeroOptimizer' object has no attribute 'defaults'
```

`min_lr` needs converting, which means asking the optimizer for its peak LR — and under DeepSpeed that object has no `.defaults`. **`min_lr_rate: 0.01` is exactly `1e-5 / 1e-3`.** ✅ R1's final LR read 1.000e-05, confirming it end to end. ⚠️ **The coupling is manual** — change `learning_rate` and you must change `min_lr_rate`.

### ⚠️ `prepared_dataset_path` is a trap at scale — leave it out

The cache saves the pre-tokenization dataset via `save_to_disk`, which ran at **~60 examples/s** in R0 — 5m33s for 20,000 documents, so **~2 hours** at R1's 500,000, inside the training job. Removing the line skips it; TRL then tokenizes at ~2,700 docs/s with `num_proc: 16`. *(Its key is `sha256({path, seed, num_samples})`, so it also silently hits if a dataset's contents change at the same path. Version dataset directories by name.)*

### ⚠️ `seed: 42` is required

`_load_dataset` does `if seed: res = res.shuffle(seed=seed)`. The train split is four parquet shards concatenated in file order, so without a shuffle `num_samples: 500000` takes essentially one crawl segment. **Always shuffle.**

### Deviations from the blog post

| Setting | Theirs | Ours | Why |
|---|---|---|---|
| `per_device_train_batch_size` | 4 | **1** | doesn't fit in 64 GB |
| `lr_scheduler_kwargs` | `min_lr` | **`min_lr_rate`** | crashes under DeepSpeed |
| `functionary_packing` | absent | **false** | inert either way — Part 4 |
| `frozen_res` norms | `norm\.weight` | anchored | see above |
| dataset | `arcee-train/afm-autodistill-mix-v0` | FineWeb-Edu | theirs is internal; ours is public and citable |
| `eval_dataset` | absent | held-out split | gives the fast metric |
| `num_train_epochs: 1` | ✓ | **`max_steps`** | a fixed token budget is comparable across configs |

## ⭐ Library versions — the policy, and why

Three environments are in play and they are not identical:

| | transformers | used for |
|---|---|---|
| `distill_env_cu126` | **5.8.1** | building students, R0, R1, all A-series tests |
| `eval-env2` | **5.9.0** | lm-eval-harness, teacher and student benchmarks |
| Arcee's reference code | **4.57** | — never run by us |

### Do NOT downgrade to 4.57 to match the reference

The whole of Track A is built on transformers 5.x and would have to be rewritten, not merely reinstalled: `@strict` dataclass configs and `PreTrainedConfig`, `create_causal_mask`, `GradientCheckpointingLayer`, `ALL_ATTENTION_FUNCTIONS.get_interface`. `modeling_afm_kda.py` subclasses **the library's own** Arcee implementation, and on 4.57 that implementation does not exist in this form. Downgrading would also re-inherit the six defects the subclassing approach was chosen to avoid, and would invalidate the runtime of a checkpoint that is already trained.

### Is differing from the reference a problem? **No — and say so explicitly in the thesis**

A model architecture is defined by its config, its weights and its arithmetic, not by a library version. What matters methodologically is only that (i) **every model being compared runs under the same version** — teacher and students both do — and (ii) the version is recorded with every number. ⭐ **A3's `0.000e+00` against the unmodified teacher, measured inside 5.x, is the evidence that our implementation is faithful.** We are reproducing their *architecture*, verified directly against the teacher, not their bytes.

The one place the gap genuinely bites is Track B: fla 0.5.2's cache API predates transformers 5.x, which is what B0 discovered.

### The 5.8.1 / 5.9.0 skew — settle it by measurement, not argument

The checkpoint carries its own `modeling_arcee.py` copied from 5.8.1, so the model *class* is pinned either way; what comes from the installed library is the mask construction, the attention interface and the cache utilities. Whether that changes logits is an empirical question with a cheap answer:

**Run `test_r1_ce.py` in both environments on the same checkpoint.** Agreement to ~1e-4 means the skew is immaterial — record both versions and move on. Disagreement means pin `eval-env2` to 5.8.1 and re-run the benchmarks. ⚠️ **Do not change `distill_env_cu126`**: it trained R1, and a working training stack is not worth risking for tidiness.

## Pre-flight — things that only bite at scale

- **Pre-warm the remote-code module cache before `srun`.** Many ranks racing to copy `.py` files into `$HF_HOME/modules/` is a real hazard, and a per-rank `HF_MODULES_CACHE` does **not** help — with `--ntasks-per-node=1` all four local processes share one `SLURM_PROCID`. The single-process `from_pretrained` before `srun` is the fix; it worked in R0 and R1.
- **~4 minutes of Triton autotuning on step 1 of every job.** Never quote `train_runtime` from a short run.
- **`HF_HUB_OFFLINE=1` is fine**, but DistillKit uses the *teacher's* tokenizer, so that cache must be complete.
- **Re-copy the `.py` files whenever `afm_kda` changes**, and clear the module cache. They are copies, not links — see R1's provenance trap.
- **`save_model` runs unconditionally** at the end of `do_distill`, writing to **`output_path`**. It does **not** copy the architecture files.
- **`eval_strategy: "no"` for any run whose timing you quote.**
- **Commit both repos before launching** — and record the *checkpoint's* hash, not the repo's. Replace the `git log` line in every run sbatch with the block below.

### ⭐ The provenance block that records the code that will actually run

```bash
echo "=== provenance ==="
git -C $WORK_LUKAS/distillkit log -1 --oneline          # the TRAINER: repo hash is correct here
echo "--- model (the code that will RUN, not the repo) ---"
cat $STUDENT/build_info.json
sha1sum $STUDENT/*.py
echo "=================="
```

`build_info.json` records the hash the checkpoint was *built* from, and the `sha1sum` lines are ground truth for the bytes that will execute — immune to build_info drifting, to a re-sync, and to a dirty tree. The trainer's own hash still comes from `git log`, because DistillKit is imported from the repo rather than copied.

## ⭐ Confirming the token count from the log itself

Do not take 300 M on trust. The run log states it twice, independently, and neither number is in `build_info.json` or any student file — both come from the final `train_runtime` line.

**Route 1 — from what the trainer measured:**

```
train_samples_per_second 12.43  ×  train_runtime 11,790 s  =  146,550 sequences
146,550 × 2,048 tokens                                     =  300.1 M tokens
```

**Route 2 — from the configuration, which the trainer never sees as a token count:**

```
max_steps 4,580  ×  global_batch 32  ×  2,048  =  300,154,880 tokens
                    (= 4,580 × 32 = 146,560 sequences)
```

⭐ **146,550 measured against 146,560 configured** — agreement to four significant figures, the residual being the rounding in `train_samples_per_second`. A third, weaker corroboration: `epoch 0.7655 × num_samples 500,000 = 382,750 documents`, which at the measured ~784 effective tokens per document is ~300 M.

📌 **"Sequences", not "documents".** With `packing: true` every one of the 146,560 rows is 2,048 real tokens, so the token count is exact. The number of *documents* consumed was larger, because packing and the 2,048 cap discard part of every long document — see the truncation figures below.

⚠️ **This check belongs in every run's write-up**, because `max_steps` is a configuration intent and `train_samples_per_second × train_runtime` is a measurement. Two routes to one number is what turns a claimed budget into a verified one.

## ⚠️ `save_total_limit: 1` is the wrong setting for the next run

R1 kept one checkpoint. For a run whose cosine schedule anneals to its floor that is defensible — the final checkpoint is the best one by construction, the held-out curve never turned up, and each checkpoint costs ~10 GB of weights (~21 GB with optimizer state).

**But it throws away the most valuable figure the run could have produced: benchmark quality against training tokens.** Loss against tokens you already have; ARC-C, HellaSwag and MMLU against tokens would answer "how many tokens does this conversion actually need?" without a single extra training run — and that is precisely the question Anamaria has to decide next.

⚠️ **`save_total_limit` keeps the most RECENT N, not a spread.** `save_steps: 500, save_total_limit: 5` on a 4,580-step run leaves you with steps 2,580–4,580 — five checkpoints all crowded at the end, which is the least useful arrangement. To get a spread, make `save_steps` large enough that every kept checkpoint survives:

```yaml
  save_steps: 1145            # 4580 / 4
  save_total_limit: 4         # -> steps 1145, 2290, 3435, 4580, all kept
```

Four checkpoints ≈ 84 GB with optimizer state. Check `df -h $WORK_LUKAS` before launching; if space is tight, three at `save_steps: 1527` is still a curve.

## Budget arithmetic

| | |
|---|---|
| Train documents | 2,921,000 |
| GPT-2 tokens (FineWeb's own column) | 3.015 B |
| AFM tokens, raw (5,000-doc sample × 970.1) | ~2.84 B |
| **Effective tokens after packing** (774/doc) | **~2.26 B** = one epoch |

At 4 nodes, `tokens/step = 65,536`, so `max_steps = target ÷ 65,536`. **Measured 25,455 tok/s at 4 nodes (R1).**

| Target | max_steps | wall-clock | local hours | share of one epoch |
|---|---|---|---|---|
| **300 M** ✅ done | 4,580 | **3.27 h** (measured) | ~420 | 13% |
| 600 M | 9,155 | ~6.5 h | ~840 | 27% |
| 1.3 B | 19,840 | **~12.5 h** | ~1,620 | 58% |

⚠️ **300 M is Arcee's number** — they report stage-1 convergence at KL ≈ 0.2 on ~300 M tokens, on their instruction mix. The larger figures are budget-shaped choices with no convergence evidence behind them.

⚠️ **Truncation, measured on 2,000 documents:** mean 1,073 AFM tokens, median 610, p95 3,293, **10.2% of documents exceed 2048** and **26.6% of tokens are lost to truncation**. `1073 × 0.734 = 787.6` against the 785 implied by R0's `epoch` — **truncation explains the entire gap** between raw and effective token counts.

⚠️ **`epoch` is measured against `num_samples`**, not the corpus. 0.7655 means 76.6% of 500,000 documents, not of 2.92 M.

---

# PART 4 — CLOSED: document isolation

*The complete story, settled. Here so it can be written up and then stop taking mental space.*

## What the problem is

Rows are 2048 tokens; FineWeb-Edu documents average ~970 AFM tokens. Rather than waste half of every batch on padding, the trainer **packs** — concatenating documents and cutting the stream into fixed rows. A row may hold the tail of A, all of B, and the start of C.

The model doesn't know that. Unless something marks the seams it reads the row as one text, so predicting B's first token may draw on A. For the nine attention layers that is cross-document attention; for the twenty-seven KDA layers the **recurrent state** flows across the boundary unless reset.

**A quality tax, not a crash.** Many well-known models are pretrained this way. The reference's published recipe does it. **Your own practical work did too** — `test_config_4B_instruct.yaml` has `packing: true` and no isolation, and produced IFEval 0.6511 loose.

## Where it goes wrong in the code

DistillKit has **two data paths**, selected by `dataset.prepacked` in `main.py`:

```python
data_collator=collate_packed_batch if config.dataset.prepacked else None,
processing_class=None   if config.dataset.prepacked else tokenizer,
```

- **`prepacked: false`** (ours). TRL's `SFTTrainer` tokenizes, packs and collates; every position is a real token, so **it emits no `attention_mask` at all**. Boundaries are destroyed at dataset-preparation time, before any model code runs.
- **`prepacked: true`.** TRL is bypassed; `collate_packed_batch` stacks whatever columns the dataset has, and `pack_logits.py` writes segment ids into `attention_mask`.

⭐ **So `functionary_packing` and `dataset.prepacked` are two halves of one feature.** `prepacked` supplies the segment ids; `functionary_packing` installs the interpreter. We enabled the interpreter and never supplied the data. **The flag's description — *"Use functionary's packing code"* — reads like "enable packing" but means "assume your data is already in functionary's format".**

## The measurements

| Probe | Question | Result |
|---|---|---|
| **A7** | Does the KDA layer honour a supplied `cu_seqlens`? | **0.000e+00** — yes, bit-exact |
| **A7b** | Does one *arrive* at the KDA layers? | **No** — fixed; now 0.000e+00 |
| **A7c** | And at the attention layers? | **No** — fixed; the monkey patch fires **zero times** |
| **R0 check 7** | Does the *pipeline* deliver a segment mask? | **No** |
| **R0b** | What does the collator send? | `input_ids` and nothing else |
| **A9** | Can the *teacher* honour a segment mask? | **No** — 1.302e+01 for both mask and control |

## Why A9 closes it

The teacher is an unmodified `ArceeForCausalLM`. Under `flash_attention_2`, transformers only builds a mask when something needs masking; a segment mask `[1,1,…,2,2,…]` contains no zeros, so it reads as "all valid" and is discarded.

**So if we packed with segment ids, the student would isolate and the teacher would not** — every token near a boundary would carry a target the student structurally cannot produce. That is worse than no isolation: a systematic, permanent error signal rather than a diffuse quality tax.

⭐ **The reframe that settles it.** As things stand, **student and teacher see exactly the same thing.** The distillation target is self-consistent, and "match the teacher" remains well posed. Path (b) would have broken that symmetry. Reviving it would mean wrapping the *teacher* the way we wrapped the student — **out of scope, by decision.**

## How much does it actually cost?

**Unknown, plausibly small, and not worth asserting either way.** At 2048 with ~970-token documents a row holds two or three fragments, so roughly half of all tokens sit downstream of a boundary. For attention, softmax is content-addressed and unrelated text generally draws low weight. For KDA the state is a fixed 20 × 128 × 128 matrix with per-head decay spread over [1, 16], so fast-decay heads flush the previous document quickly and slow-decay heads carry it.

**The second-order effect cuts the other way and matters more.** These layers are *trained*, on this exact distribution, to match a teacher reading the same contaminated rows. They learn to accommodate it. The plausible residual harm is that they learn to decay slightly faster than ideal — **a mechanism, not a measurement.**

**For the thesis:** one sentence in the methods section — trained with packing and without intra-document masking, as the reference's published recipe does; quantifying its effect on a linear-attention student is left to future work.

## What the A7 work bought anyway

Roughly fifteen lines are dead code. Keep them: A3 proves the model is bit-identical without a segment mask, and `is_segment_mask(None)` short-circuits immediately. The rest is not dead:

- **A7** proved `GradientCheckpointingLayer` forwards `**kwargs`, retiring one of the six reference defects.
- **A7b** found that **fla 0.5.2 silently overwrites a supplied `cu_seqlens` whenever a mask is present** — which fires the moment batched evaluation with padding happens, and it will (Track B, step B7).
- **A7c** measured the monkey patch at zero calls, and caught `forward` nested inside `__init__`.
- The line of inquiry produced the finding that **batched left-padded generation needs the 2-D mask routed to KDA**.

⚠️ **The process lesson.** `main.py`'s two-line use of `functionary_packing` was readable from day one and would have reordered the whole investigation. **Read how a flag is consumed before building on what its name implies.**

---

# PART 5 — TRAPS

- **⚠️ Read how a flag is consumed before building on what its name implies.** `functionary_packing` reads as "enable packing"; it means "assume the data is already packed in functionary's format". Its whole implementation is two lines.
- **⚠️ Confirm a token budget two ways.** `max_steps × global_batch × seq_len` is configuration intent; `train_samples_per_second × train_runtime × seq_len` is measurement. R1: 146,560 configured, 146,550 measured.
- **⚠️ `save_total_limit` keeps the most RECENT N, not a spread.** To get quality-versus-tokens you must widen `save_steps` so that every kept checkpoint survives to the end.
- **⚠️ A derived number sitting next to published numbers looks published.** Revision 17 first printed 76.8 % and 93.0 % in a table beside Arcee's figures; both are *our* relative-score transformation of their 48.5 and 74.3. **Print the raw numbers and the formula, and label every transformed column.**
- **⚠️ Relative score, not raw point gap.** `(student − chance) / (teacher − chance)` is RADLADS's metric; a raw gap flatters tasks with a high chance floor (WinoGrande's is 0.50).
- **⚠️ MMLU falls hardest in every attention-conversion table.** Do not read three good benchmarks as four.
- **⚠️ Evaluate one path, consistently.** `output/r1_stage1_hybrid31/` and `.../checkpoint-4580/` hold the same weights but different `.py` copies; mixing them makes the provenance line uncheckable.
- **⚠️ The remote-code module cache lives under `$HF_HOME`, not `~/.cache/huggingface`.** Clearing the wrong one is silent: the model loads and runs the old code. Proof: probe prints deleted from the repo still appeared in the next job.
- **⚠️ `save_pretrained` DOES copy remote-code `.py` files into a checkpoint.** I claimed the opposite. Check the directory before writing a workaround for a problem that isn't there.
- **⚠️ A provenance line can record the wrong thing.** R1's sbatch printed `afm_kda` at `3fb1104`; the model that trained was the checkpoint's frozen copy at `3aafffc`, proved by 48 `[AFMKDA-PROBE]` lines from prints already deleted from the repo. **Record the checkpoint's `build_info.json` hash, not the repo's `git log`.**
- **⚠️ A flat curve at the end of a cosine schedule is the schedule, not the model.** R1's held-out loss moved 0.0002 over its last 500 steps — at 1% of peak LR. Saturation and annealing look identical and mean opposite things.
- **⚠️ An annealed cosine run cannot be extended by resuming.** More tokens means a new run with a longer `max_steps` from the start.
- **Check that the reported scalar loss equals the configured weighted sum.** `0.2·CE + 0.2·KL + 0.6·hs_cosine = 0.5653` against a logged 0.5621. A silently-ignored loss weight is invisible in every other number.
- **Pre-register the pass condition.** `hs_cosine` was named as R1's first real signal *before* R1 ran, which is what makes 0.841 → 0.070 a result rather than a reading.
- **A test that supplies the input it assumes confirms the assumption, not the system.** A7 → A7b → A7c → R0 → R0b → A9: each hop moved the question closer to production and each hop found something.
- **⚠️ A probe that can only report success tells you nothing when it fails.** R0's check-7 print sat *inside* the branch it was testing. **Instrument the decision, not the happy path.**
- **⚠️ stdout and stderr are two different files under SLURM.** B0 printed `RAISED:` to stdout and its traceback to stderr, so the `.out` recorded the failure and hid the reason.
- **A patch that is installed is not a patch that runs.** Count the calls.
- **Two identical numbers are a finding**, stronger than a merely-large one. It has meant "the mask is ignored entirely" three times — A7b, the vanished `forward`, and A9.
- **⚠️ `min_lr` under DeepSpeed is an `AttributeError`, not a wrong number.** Use `min_lr_rate`, and remember it is coupled to `learning_rate` by hand.
- **⚠️ A dataset cache that helps at 20k documents can cost hours at 500k.** `save_to_disk` runs at ~60 examples/s.
- **`warmup_ratio` × a small `max_steps` is no warmup at all.** 0.025 × 20 = half a step. R0's loss spike was entirely an artifact of its length — and R1, with 114 warmup steps, never spiked.
- **The first step of a KDA job costs ~4 minutes of Triton autotuning.**
- **Know which log lines are per-rank and which are global.** `Froze 147 tensors` × 16 is a consistency check; `train_runtime` more than once means the job split.
- **DistillKit's config has two levels**, and some features have two halves.
- **DeepSpeed's `grad_accum` wins.**
- **`re.search`, not `re.match`.** Anchor every parameter-name regex.
- **`save_model` writes weights and `config.json`, not the architecture files.** A remote-code checkpoint needs the five `.py` files copied in before it will load.
- **Checkpoints carry copies of the model code.** Every fix needs a re-copy and a module-cache clear — which is also what makes editing the repo safe while a job runs.
- **A config value can become load-bearing for a reason it wasn't chosen for.** `per_device_train_batch_size: 1` was a memory decision, then a correctness precondition.
- **Library versions belong in the run log.** fla 0.5.2 and fla main differ in `cu_seqlens` handling; `distill_env_cu126` has transformers 5.8.1 and `eval-env2` has 5.9.0.
- **A single-layer benchmark is not a model benchmark.** A6's 0.31× became ~0.88× at the model level.
- **Conservative estimates are better than optimistic ones.** 3.01× predicted, 3.36× measured; 3.7 h predicted, 3.27 measured. An estimate that comes in early costs nothing; one that comes in late costs a queue slot.
- **Two independent measurements agreeing is worth more than either alone.** A5b's 7.9412 and R0's 7.918.
- **Output goes where `output_path` says**, not where the last project put it.

## Sources

- `r1_stage1_hybrid31-58141472.out` / `.err` — **the stage-1 run**
- `r0_kda_smoke-58114936` (the `min_lr` crash), `-58117838` (R0), `-58131180` (R0b)
- `eval_zs_teacher_afm45b_base-58216691` — the teacher baseline
- `main.py` / `configuration.py` / `trainer.py` / `signals.py` / `pack_logits.py` / `monkey_patch_packing.py` (DistillKit fork) — all read in full, 2026-09-18
- `test_config_4B_instruct.yaml` — the practical work's config; same packing situation, good results
- `kda_installed_0.5.2.py` — the authoritative fla source
- `blogpost_yaml.txt` — Arcee's stage-1 configuration, verbatim
- `RADLADS.pdf` — the three-stage pipeline, and the 0-shot / 5-shot-MMLU evaluation convention. ⚠️ **Says nothing about packing or document boundaries**
- `claude/building_the_student.md` — Tracks A, B and C
- `claude/startup_plan_and_pitfalls.md` — steps 5–8, the multi-node launch script, Leonardo accounting