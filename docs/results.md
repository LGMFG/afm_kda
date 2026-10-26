# Results — the current truth, one finding per entry

*Created 2026-10-12, updated 2026-10-26 (speed benchmark §8; GSM8K / code for all FineMath students §7) and 2026-10-16 (R5 MMLU/GSM8K; stage-2 training outcomes for all five runs; seed replicates; stage-2 arithmetic, `ce_outliers`, 0-shot and MMLU for stage 2 and seeds). Each finding appears **once**, with its final numbers, its status and where the history lives. When a new result changes a finding, edit the entry here and record the history in the lab notebook (`claude/first_distillkit_runs.md`, "notebook" below). Live run status: `claude/STATUS.md`.*

*Legend: ✅ established · 🔶 partial / caveat · ⬜ pending · ❌ hypothesis falsified. "rel." = relative score (student − chance)/(teacher − chance), computed by us, not published.*

---

# PART I — Instruction distillation (practical work, done)

*Source: practical-work report (`report.txt`). Qwen3-4B-Instruct-2507 → Qwen3-0.6B-Base, DistillKit online, CE 0.3 / KL 0.7, T = 2, lr 1e-4, length 4096, 1 epoch, 1 node.*

| finding | numbers | status |
|---|---|---|
| Instruction following transfers | IFEval prompt-strict: base (native format) **15.5** → Tulu **50.1** · Dolci **59.7** · teacher **83.7** (± 1.6–2.2 SE); base with chat template 26.8 | ✅ >12 SE |
| Dataset matters more than loss weighting | Dolci − Tulu = **+9.6** (≈ 3.2 SE); KL-heavy vs CE-heavy on Tulu: 50.1 vs 46.0 (≈ 1.4 SE, suggestive) | ✅ / 🔶 |
| General benchmarks do not improve | distilled students 1–5 points **below** the base on ARC-C, HellaSwag, WinoGrande, MMLU | ✅ |
| Math barely moves | GSM8K-CoT (4-shot, flexible): base 60.7 · Tulu 62.8 · Dolci 62.9 · Tulu-math 60.4 · MetaMath 64.9 · teacher 72.6 | ✅ |
| What transfers is behaviour | stopping, format, simple constraints; repetition loops: base 54.9 % → Tulu 34.4 % / Dolci 27.0 % (teacher 1.5 %); median length 1,195 → ~257 words (teacher 195) | ✅ heuristic |
| Cost | Tulu 10,075 steps / 660 M tokens / 7.7 h; Dolci 19,348 / 1.27 B / 14.7 h (1 node, 4 × A100) | ✅ |

**Lessons carried into Part II:** DistillKit set up and validated; KL-weighted objective; the evaluation regime decides the verdict; single runs give no variance estimate.

---

# PART II — Converting AFM-4.5B attention into gated delta-rule attention

## 1. Setup facts the results rest on

### Teacher and corpus

| | | source |
|---|---|---|
| Teacher | AFM-4.5B-Base: 36 layers, hidden 2560, 20 query / 4 KV heads, ReLU² MLP, 4,619,184,640 params | startup doc §5 |
| Teacher baseline validates the harness | ARC-C acc 55.63 (Arcee 55.6) · HellaSwag acc_norm 78.16 (78.0) · GSM8K 53.53 (52.1) · MMLU 65.00 (63.1 — micro vs macro average) | notebook rev 20 |
| Corpus | FineWeb-Edu `sample/100BT`, 4 shards (revision `87f09149`), 2,923,000 docs, **3.015 B GPT-2 tokens** | startup doc §3 |
| Splits | train 2,921,000 · held-out 2,000 (carved before any training) | startup doc §3 |
| Effective tokens | **784.1 tokens/doc after packing** (measured), one epoch = **2.2903 B** tokens = 1,118,292 sequences of 2048 | notebook rev 33 |
| Truncation | 10.2 % of docs exceed 2048; 26.6 % of tokens lost to truncation | notebook Part 3 |
| Stage-1 pool | `shuffle(seed=42)` → first 500,000 docs; R1–R4 consumed epoch 0.7655 of it (≈ 383 k docs) | notebook Part 3 |
| Exact duplicates in the corpus | stage-1 pool holds 1,131 internal duplicates; **9 of the 2,000 held-out texts also occur in stage 1** (0.45 %) | notebook rev 42 |
| Stage-2 data | 1,350,000 docs, seed-42 positions 500,000–1,864,858, duplicates skipped, **0 overlap** with stage 1 and held-out | notebook rev 42 |
| Packing | without document isolation, like the reference; teacher could not honour a segment mask (A9) — student and teacher see the same input | notebook Part 4 |

### The four students

| | **R1 KDA hybrid** | **R2 GDN hybrid** | **R3 KDA all-linear** | **R4 GDN all-linear** |
|---|---|---|---|---|
| linear / attention layers | 27 / 9 (attention at 3, 7, …, 35) | 27 / 9 | 36 / 0 | 36 / 0 |
| parameters | 4,999,026,076 | 5,141,828,536 | 5,046,996,688 | 5,237,399,968 |
| trainable in stage 1 | 477 tensors (≈ 945 M, 19 %) | 396 | 576 | 468 |
| frozen in stage 1 | 147 tensors in all four: embeddings, lm_head, all norms, all MLPs |
| CE at initialisation | 7.9269 (A5b: 7.9412, R0: 7.918) | 7.9247 | ≈ 8.5–8.6 (step 0) | ≈ 8.5 (step 0) |

**Shared by all four:** everything outside `self_attn` is the teacher's, byte for byte; q/k/v/o transferred from the teacher with a group-major 5× key/value repeat (GQA → MHA); attention layers keep the teacher's projections but lose RoPE (**NoPE costs +3.68 CE** at init) and gain a sigmoid output gate (**+0.58 CE**, `gate_proj` +6,553,600 params); weight transfer saves **−0.21 CE**. Model code: one file, one adapter per mechanism.

**KDA vs GDN, per linear layer** (measured): KDA **27,612,308** params in 16 tensors, GDN **32,901,288** in 13. Difference +5,288,980 = **+5,895,680** for GDN's full-rank output gate **−604,160** for its per-head (instead of per-channel) forget gate. **Matched on projection geometry and initialisation, not on parameter count.** Forget gate: KDA per channel (2,560 decay values per token via a 128-dim bottleneck) vs GDN per head (20 scalars).

**Correctness checks:** all-attention student vs teacher **0.000e+00** on logits and all 37 hidden states (A3); cached == uncached generation, the teacher shows the identical bf16 residual (B6e); batched left-padded generation correct (B7/B7c); KDA and GDN start **0.0022 nats apart**, an order of magnitude inside the ~0.023 seed spread of the init (C1).

### Stage-1 recipe (identical for R1–R4)

Arcee's merged RADLADS steps 1+2: loss **0.2 CE + 0.2 KL (T = 2) + 0.6 hidden-state cosine** (`layer_mapping: all`), LR **1e-3** cosine to 1e-5 (≈ 114 warm-up steps), sequence 2048, global batch 32 (4 nodes × 4 A100 × grad-accum 2), **4,580 steps = 300.1 M tokens** (confirmed two ways), MLPs/embeddings/norms frozen, ZeRO-2, bf16.

### Evaluation protocol (frozen)

ARC-C (acc_norm), HellaSwag (acc_norm), WinoGrande (acc) 0-shot · MMLU (acc) 5-shot · GSM8K (strict exact match) 5-shot · no chat template · `use_cache=True` for teacher and students alike · lm-eval harness, transformers 5.9.0 (training 5.8.1 — numerically identical, checked twice).

---

## 2. Stage-1 training outcomes

| run | runtime | tok/s | CE / KL / hs_cos, first microbatch → mean of last 20 | held-out composite (final) |
|---|---|---|---|---|
| **R1** KDA hybrid | 3.27 h | 25,455 | 7.986 / 9.286 / 0.844 → **2.261 / 0.314 / 0.069** | **0.5686** |
| **R2** GDN hybrid | 2.87 h | 29,082 | 7.981 / 9.349 / 0.840 → 2.283 / 0.355 / 0.070 | 0.5814 |
| **R3** KDA all-linear | 3.61 h | 23,081 | 8.590 / 10.040 / 0.837 → 2.276 / 0.334 / 0.068 | 0.5744 |
| **R4** GDN all-linear | 3.04 h | 27,402 | 8.531 / 9.914 / 0.846 → 2.303 / 0.379 / 0.075 | 0.5937 |

✅ All four clean (147 frozen ×16, monotone held-out curves, no NaN/OOM). **R1 held-out CE 2.4559 vs teacher 2.2963 → 97.2 % of the initialisation gap closed** (perplexity 11.7 vs 9.9).

**vs Arcee (same teacher, same recipe, different corpus):** KL **0.314** vs their ≈ 0.2 at 300 M tokens — the only like-for-like comparison. Their published benchmarks come from ≈ 1.3 B tokens incl. a 32k stage with MLPs unfrozen; ours is stage 1 alone (23 % of their tokens): ARC-C acc 47.78 vs 48.5 · HellaSwag 72.40 vs 74.3 · MMLU 30.55 vs 55.1 · GSM8K 2.43 vs 36.5.

---

## 3. The 2×2 at a glance

**Benchmarks, relative score:**

| | MMLU 5-shot | GSM8K 5-shot | ARC-C | HellaSwag | WinoGrande | 0-shot mean |
|---|---|---|---|---|---|---|
| **R1 KDA hybrid** | **13.9 %** | **4.5 %** | **79.0 %** | 89.2 % | 79.6 % | 82.60 % |
| R2 GDN hybrid | 11.5 % | 2.5 % | 74.1 % | 87.9 % | 71.5 % | 77.82 % |
| **R3 KDA all-linear** | 7.5 % | 1.6 % | 78.6 % | **90.1 %** | **82.3 %** | **83.64 %** |
| R4 GDN all-linear | 5.8 % | 0.1 % | 68.3 % | 87.5 % | 72.7 % | 76.17 % |

Raw: teacher 0.5922 / 0.7816 / 0.7048 / MMLU 0.6500 / GSM8K 0.5353 · R1 0.5205 / 0.7240 / 0.6630 / 0.3055 / 0.0243 · R2 0.5034 / 0.7174 / 0.6464 / **0.2959** · R3 0.5188 / 0.7289 / 0.6685 / **0.2799** · R4 0.4838 / 0.7153 / 0.6488 / **0.2730** (MMLU raw for R2–R4 added 10-16; earlier only relative scores had been recorded; ARC-C SE ±0.015, HellaSwag ±0.0045, WinoGrande ±0.013). GSM8K strict: R2 0.0136, R3 0.0083, R4 0.0008.

**Held-out (64 × 2048 tokens, one process):**

| | CE | KL (T = 2) | hs_cosine |
|---|---|---|---|
| teacher (self) | 2.2963 | 0.0000 | 0.0000 |
| R3 KDA all-linear | 2.4379 | 0.3328 | 0.0711 |
| R1 KDA hybrid | 2.4559 | 0.3568 | 0.0754 |
| R4 GDN all-linear | 2.4623 | 0.3779 | 0.0785 |
| R2 GDN hybrid | 2.5082 | 0.4357 | 0.0835 |

---

## 4. RQ-II.1 — Mechanism: KDA vs GDN

| finding | numbers | status |
|---|---|---|
| **KDA ahead on every measure, in both layouts** | 7 / 7: held-out CE, KL, three 0-shot tasks, MMLU, GSM8K | ✅ within this experiment |
| KL is the discriminating metric | GDN − KDA in held-out KL **+22.1 %** (hybrid), **+13.6 %** (all-linear); CE only +2.1 % / +1.0 % | ✅ |
| 0-shot benchmarks | +4.78 (hybrid) / +7.47 (all-linear) points of mean relative score; pooled Z ≈ 3.1 on sampling error only | ✅ / 🔶 no seed variance |
| Arithmetic (probe v2, paired) | **+13.6 [+11.9, +15.3]** (hybrid), **+10.0 [+8.2, +11.7]** (all-linear) points pooled; 2–3× on 2-digit add/sub | ✅ largest mechanism effect |
| Behaves like a per-layer property | held-out composite gap per linear layer 4.74e-4 (27 layers) vs 5.36e-4 (36) | 🔶 |
| Not an initialisation effect | 0.0022 nats apart at init | ✅ |
| **Trade-off: GDN trains faster** | 1.14× (hybrid), 1.19× (all-linear), −3.4 % memory — candidate cause GDN's fused q/k/v convolution path | ✅ measured; cause 🔶 |
| **Mechanism gap on 0-shot survives seeds** | 0-shot mean relative, KDA − GDN (hybrid, stage 1), seeds 42/43/44: +4.78 / +6.94 / +6.15 = **+5.96 ± 1.09**; seed SD of the 0-shot mean: R1 0.22, R2 1.27 points; per task ARC-C seed SD 0.009 / 0.015 raw (≈ its sampling SE 0.0146) | ✅ |
| Mechanism gap on MMLU (stage 1) is small and seed-sensitive | KDA − GDN raw, seeds 42/43/44: +0.0096 / +0.0228 / +0.0046 = +0.012 ± 0.009; MMLU seed SD 0.0052 (R1) / 0.0071 (R2) raw — larger than lm-eval's ±0.0039 | 🔶 not robust at stage 1 |
| **Seed variance (data order) is tiny — the gap is ~20× larger** | held-out composite, seeds 42/43/44: R1 0.5686 / 0.5697 / 0.5693 (SD 0.0006) · R2 0.5814 / 0.5814 / 0.5813 (SD 0.00006); paired gap R2 − R1 = 0.0128 / 0.0117 / 0.0120 → **0.0122 ± 0.0006** (t ≈ 37, n = 3) | ✅ for held-out loss; benchmarks ⬜; init variance not measured |
| **Arithmetic seed spread: SD 1.1–1.7 points; mechanism gap robust** | pooled probe v2, seeds 42/43/44: R1 38.3 / 36.8 / 38.9 · R2 24.7 / 25.2 / 27.8; KDA − GDN (hybrid) +13.6 / +11.5 / +11.1 = **+12.1 ± 1.3** | ✅ ⚠️ single-run differences below ~4 points are within run-to-run noise (SD of a difference ≈ 2 points); paired CIs only cover problem sampling |
| Train-stream KL is too noisy to rank runs | last-20 rank-0 microbatches: R1 0.314 / 0.283 / 0.310 — the seed spread (0.03) is as large as the R1–R2 gap | ✅ use held-out numbers |

*Notebook: rev 22–25, 28–29, 35; "The complete 2×2", "The loss components".*

## 5. RQ-II.2 — Layout: hybrid 3:1 vs all-linear

| finding | numbers | status |
|---|---|---|
| **No reliable effect on short-context modelling** | held-out composite effect changes **sign** between two held-out samples (+0.0058 / −0.0110 KDA); 0-shot differences within 1 SE | ✅ (null) |
| **Hybrid clearly better on 5-shot MMLU** | −2.56 points raw (4.7 σ) KDA, −2.29 (4.3 σ) GDN when going all-linear | ✅ replicated in both mechanisms |
| **Hybrid keeps the GSM8K answer format** | flexible/strict ratio 1.44× → 1.73× → 3.93× → **40.75×** (R1, R2, R3, R4); termination with `####`: teacher 98 %, R1 56 %, R2 45 %, R3 31 %, **R4 1 %** | 🔶 true after FineWeb training only — after a FineMath stage 2 all four keep the format (flex/strict 1.00–1.10, §7) |
| No layout effect on arithmetic | R1 38.3 vs R3 38.7; R2 24.7 vs R4 28.7 (pooled probe v2) | ✅ |
| Interpretation | retained attention layers matter where something must be retrieved across a long prompt | 🔶 inferred — **retrieval probe pending** |
| Inference memory (B9a, measured) | teacher 2,304 MB at 32k · hybrid **611 MB (3.77×)** · all-linear **47.1 MB (48.9×)**; break-even at **T ≈ 670 tokens** (below it the students hold MORE state); KDA state = 1.3086 MB per layer (fp32 recurrent state) | ✅ |
| Training speed | all-linear slower than hybrid (0.91× KDA, 0.94× GDN): one KDA layer is 0.31× an attention layer at 2k, crossover ~16k, 1.71× at 32k | ✅ |
| Matches Arcee | their KDA-only ≥ hybrid on MMLU/ARC-C but worse on GSM8K | ✅ partly |

*Notebook: rev 24, 28–29; "The complete 2×2 — … CONTEXT LENGTH"; `track_b.md` B9a; `building_the_student.md` A6.*

## 6. RQ-II.3 — What survives and what is lost (R1 unless stated)

| finding | numbers | status |
|---|---|---|
| Knowledge / commonsense survives | 79–90 % relative on ARC-C, HellaSwag, WinoGrande | ✅ |
| The student can use few-shot exemplars | ARC-C 0.5205 → **0.5503** at 5-shot; all five metrics improve | ✅ — falsified the "shot count" explanation |
| **MMLU collapses: 13.9 % relative** | uniform across subject areas (8.2–19.6 %); Arcee's hybrid keeps 79 % on the same letter-scored task, so the format is retainable | ✅ measured |
| MMLU explanation | **Resolved by the cloze control (rev 53):** stored knowledge is retained at 76–80 % of the teacher in every student; the teacher gains 20.7 points from the lettered multiple-choice format, the students gain nothing (they lose 5–13). What the conversion lost is that format skill — reading labelled options and emitting the right label — which the attention layers partly carry (hybrids keep more, stage 2 restores some in hybrids only). Budget with frozen MLPs ruled out (R5). Arcee's students keep the letter format (79–81 % rel.); their corpus/fine-tune is the uncontrolled difference | ✅ |
| **GSM8K: 4.5 % relative, explained by single-step arithmetic** | probe v1: teacher 76 %, R1 20 %; `0.76^k = 0.5353` → k = 2.28; predicts R1 `0.20^2.28` = **2.56 %** vs measured **2.43 %** | ✅ for R1 (fits R1, R4; not R2, R3) |
| Degeneration is a symptom | GSM8K samples (100/model): loops teacher 1 %, R1 39 %, R2 51 %, R3 32 %, R4 42 % | ✅ |
| **Arithmetic: reading intact, procedure lost** (probe v2) | copy 100 % for every model; pooled: teacher 91.0 · R1 38.3 · R2 24.7 · R3 38.7 · R4 28.7 | ✅ |
| Loss tracks the size of the problem space | student/teacher: 1-digit tables ≈ 1.0 · mul 2×1 0.84 · add 2-digit 0.77 · add 3-digit 0.10 · 4-digit 0 (R1) — memorised facts survive, the general algorithm does not | ✅ pattern; mechanism 🔶 |
| Tokenizer | 1–3-digit numbers are ONE token; 4-digit split left-to-right ('335', '7') → no digit positions to align (retracts the earlier digit-alignment story) | ✅ |
| Carry propagation is not the failure | excess carry penalty over teacher 0–7 points (< 10 pre-registered) | ❌ hypothesis |
| Errors are structured | wrong answers have the right units digit 40 % (R1) vs ~10 % chance, 75 % teacher; operand copies ≈ 1 % | ✅ |
| **Gap doubles off the training distribution** | CE gap web **+0.1595** · maths +0.3372 (**2.11×**) · instruction/code (Tulu 3) +0.3320 (**2.08×**); 2.82× with long docs only; web gap invariant (+0.1592) | ✅ ~19 σ |
| …attributable to the architecture? | would need the same recipe on a non-converted student | ⬜ limitation |
| Code (HumanEval, base protocol, 0-shot pass@1) · retrieval (only if time remains) | HumanEval: stage 1 0.012 (all four), FineWeb stage 2 0.012–0.055, FineMath stage 2 0.085–0.122 (teacher 0.250); MBPP see §7 | ✅ (retrieval not run) |

*Notebook: rev 18–20, 26–27, 30–32, 35; "RESOLVED — … ARITHMETIC CIRCUIT"; "What stage 1 did NOT recover"; "R5 diverged … what the probes found".*

## 7. RQ-II.4 — Recipe: more tokens, unfreezing, learning rate

### R5 — one full epoch of stage 1 (2.29 B tokens, 7.6× R1)

| finding | numbers | status |
|---|---|---|
| No 0-shot gain | mean relative 82.6 % → 77.8 %; gains above ~+2 points ruled out | ✅ "better" ruled out; "worse" not established |
| Held-out composite not better | 0.5715 vs R1 0.5686 | ✅ (pre-registration A failed) |
| **Training diverged twice and recovered** | eval loss 0.638 → **1.742** → 1.062 → **1.755** → 0.5715; grad-norm max **83,070**; at checkpoints 6988 and 10482 it could not copy a 3-digit number | ✅ |
| Cause: long time at high LR | steps with LR > 5e-4: R5 17,580 vs R1 2,304; every spike in that window | ✅ correlation |
| Small arithmetic gain | **+7.6 [+6.1, +9.2]** points pooled (closes 14 % of the gap); batching control: +7.4 | ✅ (pre-registered NULL falsified) |
| **MMLU unchanged** | 0.3068 vs R1 0.3055 (**14.2 % vs 13.9 % rel.**), Δ +0.0013, 95 % CI [−0.010, +0.012] — 7.6× stage-1 tokens buy nothing on MMLU | ✅ |
| **GSM8K slightly up** | strict 0.0409 vs 0.0243 (**7.6 % vs 4.5 % rel.**), Δ +0.0166, z ≈ 2.4; flexible/strict 1.09 (R1 1.44); predicted from the arithmetic gain (1.20^2.28 × 0.0243 = 0.037) — heuristic, mixes probe v1 k with v2 ratio | ✅ consistent with GSM8K-follows-arithmetic |
| **Latent, input-triggered damage** | 4/64 web and 9/64 Tulu sequences catastrophic (+3.7 to +10.3 nats); without the worst five R5 − R1 = **+0.0002**; massive activation from layer 3 (max |h| **1,408** vs 162); layer-0 KDA projections 4–6× R1's norm; `dt_bias` unchanged; sdpa vs FA2 agree | ✅ — "recovered" ≠ undamaged |
| Consequence | R5 is not a base model; a longer stage 1 is not the lever. HALO independently finds > 320 M stage-1 tokens do not help | ✅ |

### Stage 2 — RADLADS step 2 on all four (trained 2026-10-08 → 10-15; evaluation queued)

Recipe: KL only (T = 2), nothing frozen, constant 1e-5 after 380 warm-up steps, 1.0 B new tokens (zero overlap), checkpoints every 100 M. LR check on R1: cosine 5e-5 → 1e-5 (HALO's value for this model shape). ⚠️ Stage-2 `eval_loss` is **KL only** on the same 200 held-out docs — comparable across stage-2 runs, **not** with the stage-1 composite (0.5686 etc.).

| run | runtime | tok/s | tokens (measured) | held-out KL (`eval_loss`) first eval → final | train KL, mean last 20 | max grad_norm | status |
|---|---|---|---|---|---|---|---|
| **R1-S2** | 11.83 h | 23,491 | 1.0005 B | **0.3379 (100 M) → 0.2731 (1 B)**, −19 %, monotone, still falling | 0.276 | 15.95 (ep 0.944, one spike) | ✅ clean |
| **R2-S2** | 11.42 h | 24,330 | ≈ 1.0 B | 0.3726 → **0.2965**, −20.4 %, monotone, still falling | — | 18.75 | ✅ clean |
| **R3-S2** | 12.82 h | 21,668 | ≈ 1.0 B | 0.3563 → **0.2937**, −17.6 %, monotone, still falling | — | 8.46 | ✅ clean |
| **R4-S2** | 11.32 h | 24,535 | ≈ 1.0 B | 0.4022 → **0.3298**, −18.0 %, monotone, still falling | — | 9.13 | ✅ clean |
| **R1-S2-cos** (5e-5 → 1e-5) | 11.86 h | 23,429 | ≈ 1.0 B | **0.5282** → 0.2956: starts far WORSE, ends 8 % worse than constant 1e-5 | — | 11.48 | ✅ clean, ❌ worse |

*Train `loss` (= KL in stage 2), mean of the last 20 logged values, is the stable train-side number; the component column of `summarize_run.py` is 20 rank-0 microbatches and noisy.*

| stage-2 finding | numbers (held-out KL on the same 200 docs, after 1 B tokens) | status |
|---|---|---|
| **Mechanism gap survives stage 2** | GDN / KDA: hybrid 0.2965 / 0.2731 = **1.086** (1.103 at 100 M) · all-linear 0.3298 / 0.2937 = **1.123** (1.129) — barely narrows | ✅ held-out KL; benchmarks ⬜ |
| **Layout gap: hybrid better, consistently** | all-linear / hybrid: KDA **1.075**, GDN **1.112**, at all ten eval points; same direction as stage-1 `eval_loss` on these docs (0.5744 vs 0.5686) | ✅ on these 200 docs |
| Mechanism ≈ layout now | R3 (KDA all-linear) 0.2937 ≈ R2 (GDN hybrid) 0.2965 | 🔶 one seed per cell in stage 2 |
| **Not converged at 1 B** | last 100 M still −0.005 to −0.009 for all four | ✅ more tokens would help |
| **HALO's cosine 5e-5 does not transfer** | KL 0.528 at 100 M (constant: 0.338) — the higher LR first undoes stage 1, then recovers; final 0.2956 vs 0.2731 (+8 %) at equal tokens | ✅ constant 1e-5 is the better recipe here; `ce_outliers`: clean, and held-out CE also favours constant (below); benchmarks ⬜ |
| Throughput | stage 2 is 6–16 % slower than stage 1 (everything trainable); GDN 1.04× (hybrid) / 1.13× (all-linear) faster than KDA | ✅ |
| **Stage 2 improves arithmetic — modestly** (probe v2, paired vs own stage-1 parent) | R1 38.3 → **46.4** (+8.1 [+6.5, +9.6], G = 0.15) · R2 24.7 → 28.4 (+3.7) · R3 38.7 → 44.4 (+5.7) · R4 28.7 → 35.1 (+6.4); teacher 91.0 | ✅ **pre-registration met** (R1-S2 ≥ 43.3 %); but 85–94 % of the gap remains |
| Where the gain is | 2-digit add/sub and 3-digit: add_2 77 → 89.5, sub_2 55 → 80, add_3 9.5 → 20.5, sub_3 3 → 17.5 (R1); **4-digit still 0–2.5 %** (teacher 79–80), mul_2 18 → 20 | ✅ the "problem-space cliff" moves slightly, does not disappear |
| Mechanism gap does not close | KDA − GDN: hybrid **+17.9** [+16.2, +19.7] (stage 1 +13.6), all-linear **+9.3** (stage 1 +10.0) | ✅ KDA benefits at least as much from stage 2 |
| Layout within stage 2 | all-linear − hybrid: KDA −1.9 [−3.7, −0.3], GDN **+6.7** [+4.9, +8.4] | 🔶 KDA within seed noise; GDN ~3 run-SD but one seed; disagrees with held-out KL (hybrid better for both) |
| Cosine vs constant on arithmetic | +2.6 [+1.1, +4.1] for cosine (49.0 vs 46.4) | 🔶 within run-to-run noise; held-out KL favours constant (+8 %) |
| **Token curve: slow, still rising** | R1-S2 checkpoints: 38.3 (0) → 38.1 / 38.8 / 39.8 (300 M) / 42.5 / 42.6 / 40.6 / 44.6 / 42.2 / 45.4 / **46.4 (1 B)**; linear fit **+0.8 points per 100 M**, residual SD 1.2 (= seed noise) | ✅ nothing before ~300 M; at this rate the teacher's 91 % would need ~5.6 B more web tokens (if it stayed linear) |
| Stage 2 ≈ R5 on arithmetic, without the damage | R5 (2.29 B frozen-MLP stage 1): 45.9 · R1-S2 (0.3 B + 1 B full-parameter): 46.4 | ✅ neither more stage-1 tokens nor unfreezing on web text closes the gap → **supports running the FineMath stage 2** (the corpus lever) |
| **Stage 2 (constant LR) lifts MMLU in the hybrids** | raw 5-shot acc, stage 1 → 2: R1 0.3055 → **0.3581** (+5.3; rel. 13.9 → **27.0 %**) · R2 0.2959 → 0.3268 (+3.1; 11.5 → 19.2 %) · R3 0.2799 → 0.2750 · R4 0.2730 → 0.2705 (flat); cosine variant R1 → 0.3146 (+0.9). Single stage-2 runs; stage-1 seed SD 0.005–0.007 → R1 ≈ 7, R2 ≈ 4 SD of a two-run difference | ✅ real for the hybrids, recipe-sensitive (cosine barely moves it); component responsible not identified |
| **The R1-S2 MMLU gain is not a letter-bias artefact** (`mmlu_letters.py`, 14,042 items, paired) | letter-calibrated acc 30.67 → 36.62 (+5.95, gain survives and grows); mean p(gold) 0.2647 → 0.2851 (chance 0.25); right only in R1-S2 2,184 vs only in R1 1,446 (McNemar z = +12.2); **44 / 57 subjects up** (sign z = +4.4). Both models are strongly letter-biased: R1 picks B on 6 % of items (gold 24.7 %), D on 38.7 %; R1-S2 over-picks C (36.5 %); per-letter accuracy tracks the bias; 45 % of predictions change between the two | ✅ real, broad gain; MMLU at this level is dominated by format handling (letter mapping) — p(gold) barely above chance |
| ⭐ **MMLU knowledge (as cloze measures it) is retained; the letter format is what is lost** (`mmlu_continuation`, 0-shot, full answer text scored like ARC) | teacher **0.4432**; stage 1: R1 0.3976 · R2 0.3977 · R3 0.3991 · R4 0.3938 (**74–77 %** rel.); stage 2: R1 0.4052 · R2 0.4021 · R3 0.4034 · R4 0.3969 (**76–80 %**). All eight students within 1.1 points of each other — mechanism ≤ 0.7, layout ≤ 0.5, stage 2 +0.3 to +0.8 — but each 6–8 SE below the teacher. Same band as ARC-C / HellaSwag / WinoGrande. Letter format (5-shot): teacher 0.6500, students 6–27 % rel., 5–13 points below their own cloze score. Per category (R1-S2): other 90 %, social 84 %, humanities 81 %, **STEM 63 %** | ✅ students share the teacher's MLPs (frozen in stage 1), and the knowledge they hold stays accessible; what the conversion lost is the lettered-multiple-choice skill. ⚠️ Caveats: (a) letter 5-shot vs cloze 0-shot confounds format with shot count for the *letter − cloze* contrast (not for student-vs-teacher within cloze) → matched control: letter MMLU 0-shot; (b) cloze has half the teacher's dynamic range (19 vs 40 points above chance) and partly rewards answer-string plausibility, so "retained" means *as far as cloze can measure* |
| **Matched-shot controls confirm it** | Letter 0-shot: teacher 0.6330 · R1 0.2896 (10 % rel.) · R1-S2 0.3428 (24 %) · R3 0.2789 (7.5 %) → at equal shots the teacher is **+19.0** above its cloze score, students **−6 to −12** below theirs (5-shot: +20.7 / −5 to −13): the shot count was not the cause. Cloze 5-shot: teacher 0.4877, R1 0.4125 (**68 %** rel.; 0-shot 76 %) — the teacher gains 3× more from five examples (+4.5 vs +1.5). Item level (cloze 0-shot, teacher vs R1-S2): same answer on **82.6 %** of items; the student's losses concentrate in maths (elementary maths −20, high-school maths −12 points) | ✅ letter-format loss confirmed; knowledge retention 68–76 % (R1) depending on shots; weaker use of in-context examples |
| **Layout gap on MMLU widens in stage 2** | hybrid − all-linear raw: KDA 0.026 → **0.083**, GDN 0.023 → **0.056**; the all-linear students stay within ~2–3 points of chance | ✅ in our recipe; ⚠️ not an architectural limit — Arcee's KDA-only reaches 55.8 (81 % rel.) after their pipeline |
| **Vs Arcee's published students (MMLU)** | teacher 65.0 here / 63.1 theirs (micro vs macro average); hybrid **35.8 (27 % rel.)** vs their **55.1 (79 %)**; all-linear 27.5 (6 %) vs their KDA-only 55.8 (81 %); their FLP 44.0 (50 %) | ✅ still far below; their numbers include a ~1 B-token 32k fine-tune and a different corpus |
| **0-shot improves slightly everywhere** | mean relative, stage 1 → 2: R1 82.6 → **86.4** · R2 77.8 → 81.1 · R3 83.6 → 84.8 · R4 76.2 → 79.3 (each task moves ≤ 1.3 raw points, ~1 SE; same sign in 11 / 12 task × model cells) | ✅ pre-registered "within 1 point of R1" (a non-degradation check) holds — no loss of knowledge |
| Mechanism within stage 2 | 0-shot mean rel. KDA − GDN: hybrid +5.3, all-linear +5.5; MMLU hybrid +3.1 raw, all-linear +0.45 | ✅ KDA still ahead |
| **Cosine 5e-5 is clearly worse on MMLU** | 0.3146 vs constant 0.3581 (−4.4 raw, z ≈ −7.8; rel. 16.1 vs 27.0 %); 0-shot mean 84.7 vs 86.4 | ✅ with KL (+8 %) and CE: **constant 1e-5 settled**; cosine's +2.6 arithmetic is within run noise. Also shows the stage-2 MMLU gain depends strongly on the recipe |
| Grad-norm spikes are isolated and harmless | max 8–19 per run (pre-clip; every stage-2 step is clipped since the typical norm ~1.4 > 0.5); no eval rise anywhere | ✅ |
| **No latent damage in any stage-2 model** (`ce_outliers`, 64 × 2048 web held-out) | 0 sequences more than +0.5 nat above R1 stage 1; worst sequence ≤ 3.20 (R1 3.18, teacher 3.05); 0 non-finite logits; sdpa = FA2 to 1e-4; validity reproduced (teacher 2.2963, R1 2.4559). Includes the cosine run, whose KL first rose to 0.528 | ✅ (contrast R5: 4 / 64 catastrophic) |
| **Held-out web CE improves little in stage 2** | gap to teacher closed by 3–16 % (table below), while held-out KL fell 18–20 % | ✅ CE barely discriminates (as in stage 1) |
| Stage 2 is a small perturbation of stage 1 | weight drift R1-S2 vs R1: largest norm ratio 1.03× (constant) / 1.07× (cosine), mostly short-conv kernels; nothing blows up (R5: layer-0 projections 4–6×) | ✅ |
| Early-layer activations shrink | R1 stage 1 carries large residual activations at layers 3–5 (RMS 0.18–0.26, max \|h\| 42–187 on the inspected sequences); in every stage-2 model RMS there is ×0.05–0.18, late layers ×1.0–1.5 | 🔶 no teacher reference measured — meaning open |

**Held-out CE, same 64 × 2048 tokens as §3** (stage 1: `test_r5_kl.py`; stage 2: `ce_outliers`, sdpa; teacher 2.2963):

| | stage 1 | stage 2 | Δ | gap to teacher | gap closed |
|---|---|---|---|---|---|
| R1 KDA hybrid | 2.4559 | **2.4310** | −0.0249 | 0.1596 → 0.1347 | 16 % |
| R1, cosine 5e-5 | 2.4559 | 2.4432 | −0.0127 | 0.1596 → 0.1469 | 8 % |
| R2 GDN hybrid | 2.5082 | 2.5021 | −0.0061 | 0.2119 → 0.2058 | 3 % |
| R3 KDA all-linear | 2.4379 | **2.4197** | −0.0182 | 0.1416 → 0.1234 | 13 % |
| R4 GDN all-linear | 2.4623 | 2.4427 | −0.0196 | 0.1660 → 0.1464 | 12 % |

KDA < GDN in both layouts (hybrid gap +0.052 → +0.071). All-linear < hybrid on this sample again, the opposite of KL on DistillKit's 200 docs: the known sample dependence of the layout effect on short-context modelling (rev 29). ⚠️ `ce_outliers` always compares with R1 stage 1, so its per-sequence, per-position and drift sections for R2–R4 compare different architectures, not stage 2 with stage 1. Per-sequence numbers against each model's own parent would need a rerun with `R1=<own stage-1 checkpoint>`.

**Pre-registered expectations** (to check): no loss spikes ✅; no catastrophic sequences (`ce_outliers`) ✅; arithmetic pooled ≥ R1 + 5 points (≥ 43.3 %) ✅; 0-shot within 1 point of R1 ✅ (+1.0 to +1.3 raw on each task); GSM8K strict above 2.43 % ⬜.

**Stage-2 and seed benchmarks (raw; 0-shot acc_norm / acc_norm / acc, MMLU 5-shot acc; ±0.015 / ±0.0045 / ±0.013 / ±0.004):**

| model | ARC-C | HellaSwag | WinoGrande | 0-shot mean rel. | MMLU | MMLU rel. |
|---|---|---|---|---|---|---|
| teacher | 0.5922 | 0.7816 | 0.7048 | — | 0.6500 | — |
| R1 (stage 1) | 0.5205 | 0.7240 | 0.6630 | 82.6 % | 0.3055 | 13.9 % |
| R1 seed 43 / 44 | 0.5094 / 0.5034 | 0.7243 / 0.7233 | 0.6677 / 0.6709 | 82.3 / 82.2 % | 0.3049 / 0.2962 | 13.7 / 11.6 % |
| R2 seed 43 / 44 | 0.5077 / 0.4795 | 0.7125 / 0.7155 | 0.6306 / 0.6504 | 75.4 / 76.0 % | 0.2821 / 0.2916 | 8.0 / 10.4 % |
| **R1-S2** | **0.5316** | 0.7338 | 0.6756 | **86.4 %** | **0.3581** | **27.0 %** |
| R1-S2-cos | 0.5273 | 0.7330 | 0.6685 | 84.7 % | 0.3146 | 16.1 % |
| R2-S2 | 0.5137 | 0.7282 | 0.6559 | 81.1 % | 0.3268 | 19.2 % |
| R3-S2 | 0.5111 | **0.7367** | **0.6772** | 84.8 % | 0.2750 | 6.3 % |
| R4-S2 | 0.4881 | 0.7243 | 0.6622 | 79.3 % | 0.2705 | 5.1 % |

*Stage-2 rows evaluated from each run's top-level directory (final weights, = `checkpoint-15260`), seeds from their top level (= `checkpoint-4580`).*

**Generative benchmarks (teacher: GSM8K strict 0.5353 · HumanEval pass@1 0.250 · MBPP pass@1 0.332):**

| model | GSM8K strict (flex) | % teacher | flex/strict | HumanEval | MBPP |
|---|---|---|---|---|---|
| R1 (s42 / s43 / s44) | 0.0243 / 0.0356 / 0.0318 (mean 0.031, SD 0.006) | 4.5–6.7 % | 1.44 / 1.28 / 1.29 | 0.012 | 0.046 |
| R1-S2 @300 M / @500 M / 1 B | 0.0311 / 0.0356 / **0.0576** (0.0637) | 5.8 / 6.7 / **10.8 %** | 1.11 | **0.030** | **0.060** |
| R1-S2-cos | 0.0599 (0.0697) | 11.2 % | 1.16 | 0.030 | 0.062 |
| R2 (s42 / s43 / s44) | 0.0136 / 0.0205 / 0.0144 | 2.5–3.8 % | | 0.012 | 0.026 |
| R2-S2 | 0.0379 (0.0516) | 7.1 % | 1.36 | 0.049 | 0.032 |
| R3 → R3-S2 | 0.0083 → 0.0265 (0.0629) | 1.6 → 5.0 % | 3.93 → 2.37 | 0.012 → 0.055 | 0 → 0 |
| R4 → R4-S2 | 0.0008 → 0.0008 (0.0227) | 0.1 % | 40.8 → 28.4 | 0.012 → 0.012 | 0 → 0.014 |
| **R1-S1-math** (FineMath stage 1) | **0.1524** (0.1539) ±0.010 | **28.5 %** | 1.01 | **0.122** | 0.114 |
| **R1-S2-math** (FineMath stage 2 on R1) | **0.2092** (0.2100) ±0.011 | **39.1 %** | 1.00 | **0.122** | **0.146** |
| run B (FineMath stage 1 → stage 2) | **0.2373** (0.2373) ±0.012 | **44.3 %** | 1.00 | 0.116 | 0.166 |
| **R2-S2-math** (GDN hybrid) | **0.1842** (0.1873) ±0.011 | 34.4 % | 1.02 | 0.116 | 0.132 |
| **R3-S2-math** (KDA all-linear) | **0.1744** (0.1789) ±0.010 | 32.6 % | 1.03 | 0.116 | 0.074 |
| **R4-S2-math** (GDN all-linear) | **0.1183** (0.1296) ±0.009 | 22.1 % | 1.10 | 0.085 | 0.058 |

GSM8K and code stay at 0–11 % of the teacher in every FineWeb-trained student; the FineMath students reach 28–49 % (rows below the line). Stage 2 roughly doubles GSM8K for R1 (≈ 4–5 seed SD), more than the arithmetic heuristic predicts (0.038 from 1.21^2.28), mostly late (300 M → 1 B); cosine = constant here. The all-linear students still lose the answer format (flex/strict 2.4 and 28). HumanEval: all four stage-1 students solve the same 2 / 164 (0.012); stage 2 reaches 3–9 / 164 (SE ±0.013–0.018: no reliable ordering). MBPP (3-shot, `[DONE]` format): all-linear ≈ 0.

**Under FineMath stage 2 (2026-10-26):** GSM8K comes back in all four students — 22–39 % of the teacher, vs 0.1–7 % for the same students after a FineWeb stage 2 (R2 ×4.9, R3 ×6.6, R4 ×148). **The all-linear students recover the answer format** (flex/strict 1.03 / 1.10, after FineWeb 2.37 / 28.4): the FineWeb-era format loss was a data effect, not an architecture effect — unlike lettered MC (§7), which stays near chance without attention. HumanEval is flat across students (14–20 / 164, SE ±0.025). **MBPP splits by layout:** hybrids 0.146 / 0.132, all-linear 0.074 / 0.058 (z ≈ 3.7 KDA, 4.0 GDN), while the mechanism makes no difference (z ≈ 0.7 / 1.0).

**FineMath runs (2026-10-17/18; eval set `finemath_4plus_heldout`):**

| model | ARC-C | HellaSwag | WinoGrande | 0-shot mean rel. | MMLU 5-shot (rel.) |
|---|---|---|---|---|---|
| R1 (FineWeb stage 1) | 0.5205 | 0.7240 | 0.6630 | 82.6 % | 0.3055 (13.9 %) |
| **R1-S1-math** (FineMath stage 1) | 0.4676 | 0.6776 | 0.6251 | **68.4 %** | 0.3151 (16.3 %) |
| R1-S2 (FineWeb stage 2) | 0.5316 | 0.7338 | 0.6756 | 86.3 % | 0.3581 (27.0 %) |
| **R1-S2-math** (FineMath stage 2 on R1) | 0.5111 | 0.7213 | 0.6701 | 82.7 % | **0.4906 (60.1 %)** |
| run B = R1fm-S2-math (FineMath stage 2 on R1-S1-math) | 0.4991 | 0.6991 | 0.6448 | 76.0 % | 0.4657 (53.9 %) |
| R2-S2-math (FineMath stage 2 on R2) | 0.5299 | 0.7160 | 0.6677 | 83.8 % (R2-S2 81.0 %) | 0.4373 (46.8 %; R2-S2 0.3268) |
| R3-S2-math (FineMath stage 2 on R3) | 0.5256 | 0.7239 | 0.6796 | 85.8 % (R3-S2 84.8 %) | **0.3079 (14.5 %; R3-S2 0.2750)** |
| R4-S2-math (FineMath stage 2 on R4) | 0.5051 | 0.7080 | 0.6598 | 79.6 % (R4-S2 79.3 %) | **0.2990 (12.2 %; R4-S2 0.2705)** |

| finding | status |
|---|---|
| **FineMath stage 2 more than doubles letter-format MMLU retention** (27 → 60 % rel.; +13.3 raw over R1-S2) at equal tokens and recipe | ✅ the corpus is the lever for the format skill; FineMath is decontaminated against MMLU (13-gram); cloze MMLU and the letter check still to run on it |
| FineMath stage 2 costs little general ability | 0-shot mean 82.7 % = R1's 82.6 %, 3.6 below R1-S2 | ✅ mild forgetting relative to web stage 2 |
| **FineMath stage 1 hurts general ability** | 0-shot mean 68.4 % vs R1's 82.6 % (ARC-C −5.3, HellaSwag −4.6, WinoGrande −3.8 raw); MMLU ≈ R1 | ✅ aligning the linear layers on maths text transfers worse to general text |
| ⭐⭐ **FineMath brings arithmetic back — even in stage 1** (probe v2, `arith_stage2.py` v2, paired bootstrap) | pooled: R1 38.3 · R1-S2 46.4 · **R1-S1-math 68.5** (+30.3 [+28.3, +32.2] vs R1, G = 0.57) · **R1-S2-math 77.0** (+30.6 [+28.7, +32.6] vs R1-S2, G = 0.69; +38.7 [+36.8, +40.7] vs R1, G = 0.73) · teacher 91.0. The corpus effect (~30 points) is ~4× FineWeb stage 2's +8.1 and ~20× the seed SD (1.1–1.7). R1-S2-math: add_2 / sub_2 / mul_2x1 at the teacher (100 / 99.5 / 100), add_3 74.5, sub_3 88, **add_4 37, sub_4 51** (every FineWeb model ≤ 2.5; teacher 79 / 80), mul_2 43.5 (R1-S1-math 50.0 — the one condition where stage 1 beats stage 2). Copy 100 % everywhere | ✅ the "lost general procedure" (rev 35) is recoverable with the right corpus — even with the MLPs frozen; data, not architecture, limited arithmetic |
| **FineMath stage 2: letter skill up, knowledge unchanged** | `mmlu_letters.py` R1-S2 → R1-S2-math: +13.3 (calibrated +12.8), p(gold) 0.285 → **0.416**, McNemar z = +29.1, **53 / 57 subjects up**; largest gains are NOT maths (management +37, marketing +31, world history +29). Cloze 0-shot: R1-S2 0.4052 → R1-S2-math **0.4099** (+0.5; STEM 0.375 → 0.412, humanities −0.6). Now letter > cloze (+8.1), as for the teacher (+20.7) | ✅ the corpus restored the lettered-MC skill across all subjects; stored knowledge stays at ~80 % |
| FineMath stage 1: cloze MMLU 0.3856 (R1 0.3976), letter 0.3151 | general knowledge access slightly worse, like its 0-shot | ✅ |
| ⭐ **FineMath lifts code too — already in stage 1** (HumanEval 0-shot pass@1, n = 164; MBPP 3-shot pass@1, n = 500) | HumanEval: R1-S1-math **0.122** (20 / 164) and R1-S2-math **0.122** (±0.026 each) vs R1 0.012, R1-S2 0.030, best FineWeb student 0.055 → **49 % of the teacher's 0.250** (FineWeb ≤ 22 %); vs R1-S2 +0.092 (z ≈ 3.2, unpaired). MBPP: R1-S1-math 0.114 (±0.014), R1-S2-math **0.146** (±0.016) vs R1 0.046, R1-S2 0.060 (z ≈ 4.5) → **34 / 44 % of 0.332** (FineWeb ≤ 19 %). S2-math − S1-math on MBPP +0.032 (z ≈ 1.5: not established) | ✅ the third skill after arithmetic and the lettered-MC format that the corpus, not the architecture, limited. FineMath is maths web pages, not a code corpus: what transfers is plausibly exact symbol manipulation and step-by-step procedure, not code knowledge (not tested) |
| ⭐⭐ **FineMath lifts GSM8K 3.6×** (5-shot strict) | R1-S2-math **0.2092** vs R1-S2 0.0576 (+0.152, z ≈ 12) → **39 % of the teacher's 0.5353** (FineWeb students ≤ 11 %); R1-S1-math **0.1524** (28 %; R1 seeds 0.024–0.036), S2-math − S1-math +0.057 (z ≈ 3.8). flex/strict = 1.00–1.01: the answer format is fully intact. Prediction (well above 0.058) met | ✅ the largest relative gain of any intervention in the project; with arithmetic (77 %), lettered MC (60 %), HumanEval (49 %), MBPP (44 %) the FineWeb-era "generative collapse" is mostly a corpus effect |
| **Run B (FineMath stage 1 → FineMath stage 2) vs A (FineWeb stage 1 → FineMath stage 2)**: same stage 2, only the stage-1 corpus differs | log clean (cache hit, 1 harmless grad spike 9.1, 11.7 h); FineMath eval KL 0.402 → **0.3056** vs A 0.468 → **0.303** — B starts ahead, both end at the same point. 0-shot mean rel. **76.0 %** (A 82.7 %, R1-S1-math 68.4 %): ARC-C −1.2, HellaSwag −2.2, WinoGrande −2.5 raw vs A. MMLU 0.4657 vs 0.4906 (−2.5, ~4 seed SD) | ✅ a FineMath stage 1 buys nothing on maths KL after stage 2 and costs general ability that stage 2 only partly recovers → **FineWeb stage 1 + FineMath stage 2 (A) is the better recipe**. **Arithmetic: B 77.6 vs A 77.0, +0.6 [−1.1, +2.2]** — no difference (B ahead on add_4 48.5 vs 37 and mul_2 49.5 vs 43.5, behind on add_3 68 vs 74.5 and sub_4 45 vs 51); +9.1 over its start R1-S1-math. **Held-out: B is the most BOS-fragile model** (concatenated layout: web KL 2.69, maths 2.79, 33 / 40 bad windows, median maths gap +1.96) and also worse than A in training format (web KL 0.802 vs 0.355; maths 0.503 vs 0.305). `--weighted`: token-weighted maths 0.441 vs A 0.321, long documents (≥ 512 tokens) 0.434 vs 0.311, **short documents 0.709 vs 0.285** (web 0.949 vs 0.370) — B is weakest at the start of a sequence even with BOS, but weighting does not close the gap to DistillKit's own 0.3056; that discrepancy stays open (different 200 documents and ~2 documents per packed row there). **Generative: GSM8K 0.237 vs A 0.209 (+0.028, z ≈ 1.7), HumanEval 0.116 vs 0.122, MBPP 0.166 vs 0.146 (z ≈ 0.9) — no established difference**; flex = strict |
| **Arithmetic token curve, FineMath vs FineWeb stage 2** (probe v2 on all 9 checkpoints; all exact, `arith_stage2.py` v3; 100 M / 600 M rerun after the mis-named files) | R1 38.3 → **52.9 at 100 M** (+14.7; FineWeb −0.2) → 61.3 / 59.4 (200 / 300 M) → 67.2 / 68.2 (400 / 500 M) → 71.5 / 71.6 / 74.0 / 76.4 → 77.0 at 1 B. **FineMath − FineWeb at equal tokens, paired: +14.9 (100 M), +22.6 (200 M), +19.6 (300 M), +25.7 (500 M), +30.9 (600 M), +31.9 (800 M), +30.6 (1 B)** — the advantage widens, it is not a one-off jump. FineWeb's curve wobbles ±2 between checkpoints (600 M 40.6, 800 M 42.2), FineMath's too (300 M < 200 M); mul_2 stalls at 41–46 from 400 M (teacher 68, R1-S1-math 50). FineWeb stage 2 at the same points: 38.1 → 39.8 (300 M) → 46.4 (1 B). Slope after 300 M **+2.22 points / 100 M** vs FineWeb's +0.76; still rising at 1 B (last 200 M +2.8). 2-digit and mul_2x1 at the teacher within 100 M (add_2 94, mul_2x1 96); 4-digit appears only from 200 M (add_4 2 → 16, sub_4 4 → 15) and climbs to 37 / 54; sub_3 27 → 88 | ✅ two components: a fast re-alignment (+14 in the first 100 M = ~⅓ of the total gain) and slow, steady learning of the harder procedures that has not saturated at 1 B. More FineMath tokens would still help arithmetic (at a further web-KL cost, not measured). Batch-vs-single 5–8 / 8 (benign, rev 36) |

**Held-out CE / KL on BOTH corpora, every model (`heldout_kl.py`, 2026-10-18).** 64 × 2048 tokens per corpus, documents concatenated in file order (= `test_r1_ce.py`), the same sequences for every model; KL = DistillKit's definition (T = 2, × T², token mean); validity reproduced (teacher web CE 2.2963, R1 2.4559 / 0.3568). Teacher CE: web 2.2963, maths 1.5603. Gap = CE − teacher CE. #>0.5 = sequences more than +0.5 nat above the teacher (R5: 4 / 64 web).

| model | web CE | web KL | maths CE | maths KL | maths / web KL | web gap | maths gap | #>0.5 web / maths |
|---|---|---|---|---|---|---|---|---|
| R1 | 2.4559 | 0.3568 | 2.0476 | 0.9928 | 2.78 | 0.160 | 0.487 | 0 / 28 |
| R2 | 2.5082 | 0.4357 | 2.2018 | 1.2463 | 2.86 | 0.212 | 0.641 | 1 / 42 |
| R3 | 2.4380 | 0.3329 | 2.1426 | 1.1503 | 3.46 | 0.142 | 0.582 | |
| R4 | 2.4623 | 0.3779 | 2.2291 | 1.2708 | 3.36 | 0.166 | 0.669 | |
| R1-S2 | 2.4310 | 0.2930 | 1.9844 | 0.8100 | 2.76 | 0.135 | 0.424 | |
| R2-S2 | 2.5021 | 0.3879 | 2.2003 | 1.1465 | 2.96 | 0.206 | 0.640 | 2 / 41 |
| R3-S2 | 2.4197 | 0.2736 | 2.0837 | 0.9509 | 3.48 | 0.123 | 0.523 | |
| R4-S2 | 2.4427 | 0.3082 | 2.1614 | 1.0653 | 3.46 | 0.146 | 0.601 | |
| R1-S2-cos | 2.4432 | 0.3043 | 2.1031 | 0.9083 | 2.98 | 0.147 | 0.543 | |
| **R1-S1-math** | **2.8121** | **1.0088** | 2.1503 | **1.4366** | 1.42 | **0.516** (median 0.314) | 0.590 (median **0.140**) | **6 / 14**, worst +5.39 / +4.44 |
| **R1-S2-math** | 2.5225 | 0.4564 | **1.7269** | **0.4420** | **0.97** | 0.226 | **0.167** | 2 / 1, worst +0.53 / +0.66 |

| finding | status |
|---|---|
| **The FineWeb students are ~3× further from the teacher on maths text than on web text** (maths/web KL 2.8–3.5 in all nine) | ✅ the conversion matches the teacher where it was trained; off-distribution the gap triples — the KL-side picture of the lost arithmetic / format skills |
| FineWeb stage 2 lowers maths KL by the same fraction as web KL (R1 −18 / −18 %, R3 −18 / −17 %, R4 −18 / −16 %, **R2 only −11 / −8 %**) | ✅ stage 2 improves proportionally everywhere, it does not specialise; the GDN hybrid gains least (as in CE, rev 48) ⚠️ in training format (table below) all four lose the same −20 / −17 %: R2's smaller gain was a concatenated-layout artefact |
| **KDA < GDN in all 8 pairings** (2 layouts × 2 stages × 2 corpora) | ✅ the mechanism effect holds off-distribution too |
| **Layout is corpus-dependent** (paired over the 64 sequences, `heldout_kl_seqs.py --pairs`; concatenated layout; z = mean per-sequence KL difference / its standard error, SE = SD of the 64 differences / √64; \|z\| > 2 ≈ p < 0.05; covers sequence sampling, not run-to-run variance): on maths the KDA hybrid beats all-linear — R3 − R1 **+0.158 ± 0.034 (z +4.7)**, R3-S2 − R1-S2 **+0.141 ± 0.028 (z +5.0)**, hybrid better on 50–52 / 64; on web all-linear is slightly ahead — −0.024 ± 0.010 (z −2.4) / −0.019 ± 0.006 (z −3.2), on 47–52 / 64. DistillKit's 200 web docs gave the hybrid +7.5 % | ✅ on harder, off-distribution text the 9 attention layers buy ~15 % KL; on in-distribution web the layouts are within ~7 %, sign sample-dependent (rev 29). Fits the arithmetic / GSM8K format failures of the all-linear students ⚠️ **superseded** by the training-format table below: there the hybrid wins on web too (+4–8 %), as in DistillKit's own eval. The web sign flip was the missing-BOS artefact (all-linear students have no softmax attention, hence no first-token sink to lose) |
| Cosine 5e-5 loses again: +4 % web, **+12 % maths** KL vs constant | ✅ constant 1e-5 settled, more clearly off-distribution |
| **FineMath stage 2 trades web for maths:** maths KL 0.810 → **0.442 (−45 %)**, CE gap 0.424 → 0.167 nat (60 % closed); web KL 0.293 → **0.456 (+56 %; +28 % vs R1)**, CE gap 0.135 → 0.226. Per-sequence clean (worst +0.53 / +0.66) | ✅ real forgetting on web text in KL/CE terms — while 0-shot drops only 3.6 rel. points and cloze MMLU is flat: held-out KL overstates benchmark forgetting. A mixed corpus is the obvious remedy (not run) |
| ⚠️ **R1-S1-math has a heavy tail**: mean web gap 0.516 vs median 0.314; maths mean 0.590 vs median **0.140**; 6 / 14 sequences > +0.5 nat, worst **+5.4 / +4.4** (`heldout_kl_seqs.py`: KL median web 0.597, **maths 0.480** vs mean 1.437). The bad sequences are bad **only for R1-S1-math** (R1, R1-S2, R1-S2-math: 0 / 6 web above +0.5; on its 14 bad maths sequences R1-S2-math's gap is +0.07 to +0.31). Its worst web sequences are ordinary prose (NIOSH, catfish, mammal evolution, homeopathy); worst maths ones LaTeX, a reference list, a lettered grade-2 question. DistillKit's final `eval_loss` (0.4392) is the stage-1 composite on 200 other documents, so it never showed this | ✅ real, model-specific |
| **`ce_outliers` on R1-S1-math (web): NOT R5's mechanism** | sdpa = FA2 (2.8121 / 2.8123), 0 non-finite rows; worst sequence CE **7.96** (teacher's worst 3.05) — a whole 2048-token sequence near-broken; without the worst five, R1-S1-math − R1 = **+0.159** (= its weaker alignment on web text, not the tail; R5: +0.0002); per-position excess **flat** (+0.33 to +0.38). But **no massive activation** (on seq 29 R1-S1-math's max \|h\| is 1–12 where R1 carries ~60 from layer 3 — it lacks R1's large early activations, as every stage-2 model does, rev 48) and **weight drift ≤ 1.23× R1** (R5: 4–6× in layer 0), `dt_bias` unchanged, 147 frozen tensors bit-identical | ✅ an input-triggered failure without a divergence footprint; shares 2 / 6 web and 6 / 14 maths bad windows with R5; triggered by a missing BOS (two rows below) |
| R5 on the two-corpus table | web KL 0.825, **maths KL 2.064** (31 / 64 maths sequences > +0.5 nat, worst +7.3) — worst model on maths | ⚠️ in the concatenated layout only — see the next row |
| ⭐⭐ **The tails vanish when the input looks like training** (`heldout_kl.py` v2, `LAYOUT=docs`: one document per sequence from its start, `[BOS] text [EOS]`, first 64 documents per corpus; teacher CE web 2.3000, maths 1.4563) | **R1-S1-math:** maths KL 1.437 → **0.430**, CE gap 0.590 → **+0.027** (closest of all to the teacher on maths), 0 sequences > +0.5 (was 14) — the 64 documents cover roughly windows 0–43 of the concatenated stream, where 11 of the 14 bad maths windows lie (estimated from mean document length; on web only 2 of 6, windows 7 and 29, so the web evidence is weaker); web KL 0.657, 4 > +0.5, worst +1.18 (was +5.39). **R5:** web KL **0.307 = R1's 0.304**, worst +0.30, 0 > +0.5 (was 4); maths 1.026 ≈ R1 1.001. R1: web 0.304, maths 1.001; R1-S2-math: web 0.355, maths 0.305, clean | ✅ R1-S1-math's and R5's catastrophic sequences are triggered by input that DistillKit never produced — a window starting mid-document and/or a document junction without EOS — not by the text itself. On training-format input both behave like normal students. ⚠️ Qualifies rev 37 (R5 "unusable"): R5's damage is a fragility to this format shift. Trigger identified in the next row |
| ⭐⭐ **The trigger is a sequence that does not start with BOS** (`heldout_kl.py` v3, one change at a time to the concatenated stream) | `concat_bos` (every window starts with BOS; junctions still without EOS): **tails gone** — R1-S1-math web 1 / maths 0 sequences > +0.5 (worst +0.63 / +0.46), R5 0 / 24 (≈ R1's 0 / 21), R5 web KL 0.352 ≈ R1 0.327. `concat_eos` (EOS at every junction, windows still start mid-text): **tails stay** — R1-S1-math 11 / 9 (worst +3.47 / +1.97), R5 4 / 29 (worst +6.2 / +7.3). Teacher, R1, R1-S2-math: no tail in any layout | ✅ R1-S1-math and R5 break on some windows that start mid-text without BOS; junctions are harmless. Plausible mechanism (not tested): the attention-sink / first-token role — R1 carries the usual large first-layers activation (~60) on such a window, R1-S1-math does not (ce_outliers, seq 29), R5 an excessive one (rev 37). Relevance: normal prompting starts with BOS; streaming / sliding-window use would not |
| R1-S1-math matches the teacher worse on web text (training format) — **not forgetting**: its converted layers were only ever aligned on maths (stage 1 starts from the converted, untrained-linear init, like R1) | docs-layout web CE gap +0.30 (R1 +0.12, R1-S2-math +0.16), KL 0.657 (R1 0.304, R1-S2-math 0.355) | ✅ aligning on maths transfers less to web text → explains its lower 0-shot (68 % vs R1's 83 %). Real *forgetting* (aligned on web, then trained on maths) applies only to R1-S2-math: web KL 0.304 → 0.355 in training format |

**Held-out CE / KL in TRAINING FORMAT — the table of record for the thesis** (`heldout_kl.py` v2 `LAYOUT=docs`, 2026-10-19: the first 64 documents of each held-out set, each one sequence `[BOS] text [EOS]` cut at 2048, as DistillKit trains and evaluates; per-document means; teacher CE web 2.3000, maths 1.4563). The concatenated table above stays as a robustness test (sequences starting mid-text without BOS).

| model | web CE | web KL | maths CE | maths KL | maths / web | #>0.5 web / maths |
|---|---|---|---|---|---|---|
| R1 | 2.4238 | 0.3044 | 1.9370 | 1.0012 | 3.29 | 0 / 22 |
| R2 | 2.4393 | 0.3329 | 1.9980 | 1.0874 | 3.27 | 0 / 33 |
| R3 | 2.4314 | 0.3171 | 2.0402 | 1.1889 | 3.75 | 0 / 40 |
| R4 | 2.4563 | 0.3608 | 2.1176 | 1.3095 | 3.63 | 0 / 46 |
| R1-S2 | 2.4051 | **0.2408** | 1.8724 | 0.8326 | 3.46 | 0 / 18 |
| R2-S2 | 2.4341 | 0.2643 | 1.9379 | 0.9129 | 3.45 | 0 / 23 |
| R3-S2 | 2.4154 | 0.2545 | 1.9809 | 0.9854 | 3.87 | 0 / 33 |
| R4-S2 | 2.4368 | 0.2850 | 2.0452 | 1.0903 | 3.83 | 0 / 39 |
| R1-S2-cos | 2.4194 | 0.2553 | 1.9916 | 0.9451 | 3.70 | 0 / 29 |
| R5 | 2.4205 | 0.3068 | 1.9746 | 1.0257 | 3.34 | 0 / 29 |
| R1-S1-math | 2.6006 | 0.6566 | **1.4829** | 0.4300 | 0.65 | 4 / 0 |
| **R1-S2-math** | 2.4600 | 0.3549 | 1.5461 | **0.3045** | 0.86 | 0 / 0 |
| R2-S2-math | 2.5300 | 0.4450 | 1.6020 | 0.3756 | 0.84 | 1 / 0 |
| R3-S2-math | 2.4798 | 0.3794 | 1.5937 | 0.3879 | 1.02 | 0 / 0 |
| R4-S2-math | 2.5071 | 0.4268 | 1.6112 | 0.4227 | 0.99 | 1 / 1 |
| run B (R1fm-S2-math) | 2.7256 | 0.8024 | 1.6464 | 0.5029 | 0.63 | 14 / 1 |

| finding (training format) | status |
|---|---|
| Matches DistillKit's own eval where comparable: GDN/KDA web after stage 2 1.098 (hybrid) / 1.120 (all-linear) vs DistillKit 1.086 / 1.123; all-linear/hybrid 1.057 (KDA) / 1.078 (GDN) vs 1.075 / 1.112 (same order, same direction); R1-S2-math maths 0.3045 vs 0.303 | ✅ this layout reproduces the training-time eval → use it in the thesis |
| **Stage 2 (FineWeb) is uniform**: web −21 / −21 / −20 / −21 %, maths −17 / −16 / −17 / −17 % for R1–R4 | ✅ no student profits more or less from stage 2 |
| **Mechanism: KDA < GDN in all 8 pairings**, GDN +9–14 % on web, +9–11 % on maths | ✅ RQ-II.1, robust |
| **Layout: hybrid < all-linear in all 8 pairings**, small on web (+4–8 %), large on maths (+18–20 %). Paired over the 64 documents (`--pairs`, 2026-10-19), all-linear − hybrid: KDA web +0.013 (z +4.5) / +0.014 (z +5.2, stage 2), maths +0.19 (z +4.1) / +0.15 (z +5.0); GDN web +0.028 (z +9.0) / +0.021 (z +7.1), maths +0.22 (z +5.6) / +0.18 (z +6.0); the hybrid is better on 48–60 of 64 web and 58–64 of 64 maths documents. Mechanism, GDN − KDA: web z +9.6 to +13.7, maths z +8.1 to +11.4, KDA better on 60–63 of 64 web documents | ✅ RQ-II.2: the 9 attention layers matter most on harder, off-distribution text; replaces the concatenated-layout reading |
| Cosine 5e-5: +6 % web, +14 % maths vs constant | ✅ |
| R5 = R1 in training format (0.307 vs 0.304; 1.026 vs 1.001) | ✅ 7.6× the stage-1 tokens bought nothing on held-out KL |
| **FineMath stage 2 trades web for maths in all four students** (training-format KL vs the same student's FineWeb stage 2): web +47 / +68 / +49 / +50 %, maths −63 / −59 / −61 / −61 % (R1 / R2 / R3 / R4). Within FineMath stage 2, GDN is +25 / +23 % behind KDA (FineWeb stage 2: +10 / +10 %) | 🔶 one seed. ⚠️ DistillKit's own FineMath eval (token-weighted, packed rows) gives GDN/KDA only **1.08** (R2-S2-math 0.3279 vs R1-S2-math 0.303, FineWeb stage 2: 1.086) — the 1.23 here is per-document and may be inflated by a start-of-sequence weakness as for run B (R2-S2-math is also BOS-fragile); check with `--weighted`. R4-S2-math DistillKit eval 0.4222 → all-linear/hybrid (GDN) **1.29** on maths (FineWeb stage 2, web: 1.11). R3-S2-math failed, rerun |
| R2-S2-math is clean in training format but BOS-fragile in the concatenated layout (14 / 24 bad windows; R1-S2-math 2 / 1) | 🔶 FineMath stage 2 can introduce the fragility; run B has it worst |
| **Per-document vs token-weighted (`--weighted`, FineMath + FineWeb stage-2 models)**: the robust models barely change (≤ 0.04 KL); R2-S2-math and run B are worse on SHORT documents (web: R2-S2-math 0.50 short vs 0.41 long, run B 0.95 vs 0.71; every other model ±0.03), so their per-document means are inflated. On maths, long documents are harder for every model, so token-weighting raises the all-linear students (R3-S2-math 0.388 → 0.423). Token-weighted maths ratios: layout all-linear/hybrid **1.32 (KDA), 1.23 (GDN)**; mechanism GDN/KDA 1.17 (hybrid), 1.09 (all-linear) — closer to DistillKit's 1.28 / 1.29 and 1.08 / 1.09 than the per-document 1.27 / 1.13 | ✅ **report token-weighted KL in the thesis** (= how DistillKit and standard LM evaluation average); per-document only as the robustness view |
| **Paired significance under FineMath stage 2** (`--pairs`, per-document, 64 documents): layout KDA all-linear − hybrid web +0.025 (z +7.0, hybrid better on 56 / 64), maths +0.083 (z +10.0, 63 / 64); GDN web −0.018 (z −2.2, all-linear better on 36 / 64 — R2-S2-math's start-of-sequence weakness), maths +0.047 (z +4.1, 46 / 64). Mechanism GDN − KDA: hybrid +0.090 / +0.071 (z +14 / +13, KDA better on **64 / 64** both corpora), all-linear +0.048 / +0.035 (z +11 / +15, 63–64 / 64). Corpus (FineMath − FineWeb stage 2, same student): web **+0.11 to +0.18** (z +6 to +8, worse on 62–64 / 64), maths **−0.53 to −0.67** (z −10 to −11, better on **64 / 64** for all four) | ✅ mechanism the most robust effect in the project; layout clear for KDA and on maths for GDN (GDN web sign is the R2 artefact); FineMath improves every maths document and costs on almost every web document |

### FineMath stage 2 on all four students (2026-10-21; R3-S2-math rerun complete)

Each cell: FineMath stage 2 (FineWeb stage 2 in brackets), same start, recipe and tokens.

| | R1 KDA hybrid | R2 GDN hybrid | R3 KDA all-linear | R4 GDN all-linear |
|---|---|---|---|---|
| arithmetic pooled | **77.0** (46.4) | **67.5** (28.4) | **72.3** (44.4) | **66.8** (35.1) |
| MMLU letter, 5-shot | **0.491** (0.358) | **0.437** (0.327) | **0.308** (0.275) | **0.299** (0.271) |
| MMLU cloze, 0-shot | 0.410 (0.405) | 0.410 (0.402) | 0.406 (0.403) | 0.402 (0.397) |
| 0-shot mean rel. | 82.7 % (86.4 %) | 83.8 % (81.0 %) | **85.8 %** (84.8 %) | 79.6 % (79.3 %) |
| GSM8K strict, 5-shot | **0.209** (0.058) | **0.184** (0.038) | **0.174** (0.027) | **0.118** (0.001) |
| GSM8K flex / strict | 1.00 (1.11) | 1.02 (1.36) | 1.03 (2.37) | 1.10 (28.4) |
| HumanEval / MBPP | 0.122 / 0.146 | 0.116 / 0.132 | 0.116 / 0.074 | 0.085 / 0.058 |
| DistillKit FineMath eval KL (final) | 0.303 | 0.328 | 0.388 | 0.422 |
| held-out KL web / maths, training format (per doc) | 0.355 / 0.305 | 0.445 / 0.376 | 0.379 / 0.388 | 0.427 / 0.423 |
| windows without BOS (bad web / maths) | 2 / 1 | 14 / 24 | 0 / 2 | 0 / 2 |

| finding | status |
|---|---|
| **The corpus effect holds for all four students**: arithmetic +30.6 (R1), **+39.1** (R2), **+27.9** (R3), **+31.7** (R4) points over the same student's FineWeb stage 2; pre-registered ≥ +15 ✅ in all four | ✅ not a KDA-hybrid peculiarity |
| **Mechanism under FineMath: KDA − GDN +9.5 [+7.6, +11.4] (hybrid) and +5.5 [+3.7, +7.3] (all-linear)** on arithmetic — about half the FineWeb gaps (+17.9 / +9.3), both above the ~4-point noise floor (pre-registered ✅). DistillKit eval KL GDN/KDA 1.08 / 1.09 (FineWeb 1.086 / 1.123) | ✅ part of KDA's arithmetic advantage was data-limited; a real mechanism effect remains in both layouts |
| **Layout under FineMath on arithmetic: all-linear − hybrid −4.7 [−6.5, −2.9] (KDA) and −0.7 [−2.6, +1.3] (GDN)** (FineWeb: −1.9 / +6.7). R3-S2-math reaches 72.3 = 94 % of R1-S2-math's 77.0, with 4-digit add/sub 33.5 / 44 (R1-S2-math 37 / 51) | ✅ arithmetic comes back largely without attention; the attention layers add a few points for KDA only. Contrast with lettered MC, which does not come back without them |
| ⭐⭐ **Lettered-MC MMLU needs the attention layers — confirmed for both mechanisms**: FineMath lifts it in both hybrids (R1 +13.3, R2 **+11.1** raw) but barely in either all-linear student (R3 **+3.3**, 14.5 % rel.; R4 **+2.9**, 12.2 %), although the all-linear students keep the same cloze knowledge (0.406 / 0.402 vs 0.410) and the best 0-shot of all (R3-S2-math 85.8 %) | ✅ a clean dissociation in 2 × 2: same knowledge, same general ability — only the in-context option→letter mapping is missing without softmax attention. Arithmetic (R3-S2-math 72.3, R4-S2-math 66.8) and the GSM8K answer format (flex/strict 1.03 / 1.10) do come back without attention — only the lettered-MC mapping does not |
| Cloze knowledge does not move with the corpus in any student (0.40–0.41) | ✅ |
| **GSM8K under FineMath: R1 0.209 > R2 0.184 ≈ R3 0.174 > R4 0.118** (SE ±0.009–0.011; unpaired z). Mechanism KDA − GDN: hybrid +0.025 (z ≈ 1.6), **all-linear +0.056 (z ≈ 4.1)**. Layout hybrid − all-linear: **KDA +0.035 (z ≈ 2.3), GDN +0.066 (z ≈ 4.7)**. Corpus: ×3.6 (R1) to ×148 (R4) over FineWeb stage 2 | ✅ same ordering as KL and arithmetic. The layout costs more on GSM8K than on arithmetic (R4 ≈ R2 on arithmetic, 66.8 vs 67.5, but 0.118 vs 0.184): multi-step reasoning over a ~1k-token 5-shot prompt seems to use the attention layers beyond single-step arithmetic (interpretation, not tested) |
| **Code under FineMath: HumanEval does not depend on layout, MBPP does.** HumanEval 0.122 / 0.116 / 0.116 / 0.085 (SE ±0.025, no ordering; R3 = R2). MBPP hybrid − all-linear: **KDA +0.072 (z ≈ 3.7), GDN +0.074 (z ≈ 4.0)**; KDA − GDN +0.014 / +0.016 (z ≈ 0.7 / 1.0). Corpus: R3 MBPP 0 → 0.074, R4 0.014 → 0.058 | ✅ layout effect on MBPP, none from mechanism. **Sample analysis (`mbpp_samples.py`, 500 problems each) — the name hypothesis is falsified, the failure is termination and degeneration:** function name copied correctly teacher 100 %, hybrids 99 %, all-linear 92–93 % (small); answer starts a new `[BEGIN]` example instead of `[DONE]` 0 / 0 / 0.2 / **17 / 26 %** (teacher / R1 / R2 / R3 / R4 = 0 / 0 / 0.2 / 16.6 / 25.6); cut off by the token limit 2 / 3.6 / 6.2 / **24 / 50 %**; repetition loop 1.8 / 2.0 / 4.4 / **25 / 54 %**; valid Python 98 / 93 / 90 / 73 / 46 %. Cutting each answer at its first `[BEGIN]` lifts the all-linear students only to 9.6 / 9.2 % (from 7.4 / 5.8; hybrids unchanged 14.6 / 13.2) → the missing end marker explains ≈ ½ of the gap for R4 and ≈ ⅓ for R3; the rest is loops and weaker code. ⭐ **Verbatim copying marker:** the 3 in-context examples use `\r\n` line endings; the teacher reproduces them in 89 % of answers, R2 68 %, R1 33 %, **both all-linear students in 0 of 500** — a formatting detail with no effect on correctness that can only come from looking back at the prompt. The hybrid − teacher gap (14.6 vs 33.2) is code correctness, not format (hybrids stop and parse like the teacher) |
| **The all-linear students keep the GSM8K answer format under FineMath** (flex/strict 1.03 / 1.10; `####` format intact) | ✅ revises §5 "hybrid keeps the GSM8K answer format": that was a FineWeb-corpus effect. What still needs attention is lettered MC, not the output format |
| Layout on the FineMath eval (DistillKit, token-weighted): all-linear/hybrid **1.28 (KDA, 0.388 / 0.303) and 1.29 (GDN)** — the same layout cost for both mechanisms; mechanism GDN/KDA 1.08 (hybrid) and **1.09** (all-linear, 0.422 / 0.388) | ✅ both effects survive a FineMath stage 2 at the same size as after FineWeb (1.075 / 1.112 layout, 1.086 / 1.123 mechanism on web) |
| R3-S2-math rerun reproduces the crashed run: eval 0.5893 / 0.5280 / 0.4909 / 0.4629 / 0.4428 vs 0.5892 / 0.5318 / 0.4892 / 0.4611 / 0.4443 at 100–500 M (identical YAML and seed) | ✅ run-to-run nondeterminism ≤ 0.004 KL — far below every effect reported |
| 0-shot: FineMath stage 2 costs R1 −3.7 rel. points, not R2 (+2.8) or R4 (+0.3). Per task ARC-C up for the GDN students, HellaSwag down by 1–1.6 raw in all three | ✅ general ability largely kept; HellaSwag (web-narrative continuation) is the one consistent cost |
| Layout on maths KL: DistillKit eval R4/R2 = **1.29**, training-format per-document only 1.13 — R2-S2-math is BOS-fragile (14 / 24 bad windows). For KDA the two agree (R3/R1: DistillKit 1.28, per-document 1.27), because R1-S2-math is robust. **No all-linear student is BOS-fragile (R3-S2-math 0 / 2, R4-S2-math 0 / 2); two of the three FineMath-stage-2 hybrids are (R2-S2-math, run B)** | ✅ consistent with the first-token / attention-sink explanation (only softmax attention has a sink to lose); per-document ratios are reliable only for robust models — `--weighted` / `--pairs` owed over all FineMath models |

### Summary of stage 2 and the FineMath runs (2026-10-19; four-student FineMath results above, 10-26)

| | R1 | R1-S2 | R1-S1-math | **A = R1-S2-math** | B = R1fm-S2-math | teacher |
|---|---|---|---|---|---|---|
| corpus stage 1 → stage 2 | web → — | web → web | maths → — | **web → maths** | maths → maths | |
| 0-shot mean rel. | 82.6 % | 86.4 % | 68.4 % | 82.7 % | 76.0 % | 100 % |
| MMLU letter (rel.) | 0.306 (14 %) | 0.358 (27 %) | 0.315 (16 %) | **0.491 (60 %)** | 0.466 (54 %) | 0.650 |
| MMLU cloze | 0.398 | 0.405 | 0.386 | 0.410 | 0.399 | 0.443 |
| arithmetic pooled | 38.3 | 46.4 | 68.5 | 77.0 | **77.6** | 91.0 |
| GSM8K strict | 0.024 | 0.058 | 0.152 | 0.209 | **0.237** | 0.535 |
| HumanEval / MBPP | 0.012 / 0.046 | 0.030 / 0.060 | 0.122 / 0.114 | 0.122 / 0.146 | 0.116 / 0.166 | 0.250 / 0.332 |
| held-out KL web / maths (training format) | 0.304 / 1.001 | **0.241** / 0.833 | 0.657 / 0.430 | 0.355 / **0.305** | 0.802 / 0.503 | 0 |
| robust to sequences without BOS | ✅ | ✅ | ❌ | ✅ | ❌❌ | ✅ |

- **Stage 2 (RADLADS step 2: full-parameter KL, constant 1e-5, 1 B tokens) is safe and uniform but, on web text, modest.** Clean logs and per-sequence checks on all four students; held-out KL −20 / −21 % (web) and −16 / −17 % (maths) for every student; 0-shot +1 to +4 points; letter-MMLU up only in the hybrids (format skill — cloze moves ≤ 0.8); arithmetic +4 to +8 (≤ 15 % of the gap); GSM8K ×2.4 for R1; code to 3–9 / 164. Mechanism and layout gaps survive unchanged (KDA < GDN, hybrid < all-linear in 8 / 8 KL pairings). Cosine 5e-5 is worse than constant 1e-5. Still improving at 1 B.
- **The corpus decides which capabilities come back.** The same stage 2 on FineMath (A) instead of FineWeb: arithmetic 46 → 77, GSM8K ×3.6, letter-MMLU 27 → 60 %, HumanEval ×4, MBPP ×2.4 — at the cost of web fidelity (web KL +47 %) and a little general ability (0-shot −3.7 rel. points, ≈ R1). Stored knowledge (cloze) does not move with the corpus in any run (0.39–0.41).
- **Where the domain corpus goes matters.** In stage 1 (R1-S1-math) FineMath buys arithmetic fast even with frozen MLPs, but leaves the converted layers poorly aligned on general text (0-shot 68 %) and fragile at the start of a sequence. Adding a FineMath stage 2 on top (B) reaches A's procedural skills (arithmetic 77.6, GSM8K 0.237, code ≈ A) but not A's general ability (0-shot 76 vs 83 %, letter-MMLU 0.466 vs 0.491), and the fragility grows. **Recipe of record: general corpus for stage 1 (alignment), domain corpus for stage 2.**
- **Under that recipe the 2 × 2 orders the same way on every measure** (R1 > R2 ≈ R3 > R4 on GSM8K 0.209 / 0.184 / 0.174 / 0.118; arithmetic 77.0 / 67.5 / 72.3 / 66.8; FineMath KL 0.303 / 0.328 / 0.388 / 0.422). The one capability that needs softmax attention is lettered MC; arithmetic and the GSM8K format come back without it.
- **Efficiency (§8): the price of the attention layers is speed at long context.** At 128k prefill the hybrids are 2.0× and the all-linear students 2.9× faster than the teacher; the KV cache shrinks 3.9× (hybrid) or to a constant 57 MiB (all-linear). Below ~10k tokens, and at batch-1 decode, the converted students are slower in this (HF eager) implementation.

## 8. RQ-II.5 — Efficiency

| | | status |
|---|---|---|
| Inference memory vs context | see §5 (3.77× / 48.9× at 32k, break-even 670 tokens) | ✅ |
| Prefill / decode speed vs context and batch | see below (job 61585206; batch 32 / 64: debug jobs 62299405/06/10/11 + teacher) | ✅ batch 1, 8, 32, 64 |

**Protocol** (`speed_bench.py` v1.2): one A100-SXM-64GB, exclusive node, bf16, HF eager forward + fla 0.5.2 kernels, teacher with FlashAttention-2; random token ids after BOS (speed does not depend on content); CUDA-event timing, median of 10 after 3 warm-ups. Prefill = one forward pass at batch 1, last-position logits only. Decode = ms per generated token over 32 greedy steps after a chunked prefill; cache = allocated memory of the KV cache / recurrent state. Stage-1 checkpoints stand for every run of the same architecture. Checks: chunked = one-shot prefill (max logit diff 0.13–0.23, argmax identical for all five), drift recheck at T = 8,192 within 0.1 %, timing spread < 1 %.

**Prefill, batch 1 — ms (speed-up over the teacher):**

| T | teacher | R1 KDA hybrid | R2 GDN hybrid | R3 KDA all-linear | R4 GDN all-linear |
|---|---|---|---|---|---|
| 128 | 31 | 77 (0.40×) | 62 (0.49×) | 92 (0.33×) | 73 (0.42×) |
| 2,048 | 98 | 111 (0.88×) | 109 (0.89×) | 115 (0.84×) | 113 (0.86×) |
| 8,192 | 396 | 417 (0.95×) | 411 (0.96×) | 424 (0.93×) | 416 (0.95×) |
| 16,384 | 850 | 822 (1.03×) | 810 (1.05×) | 813 (1.05×) | 798 (1.07×) |
| 32,768 | 2,057 | 1,710 (1.20×) | 1,686 (1.22×) | 1,595 (1.29×) | 1,565 (1.31×) |
| 65,536 | 5,756 | 3,828 (1.50×) | 3,777 (1.52×) | 3,184 (1.81×) | 3,127 (1.84×) |
| 131,072 | 18,366 | 9,371 (**1.96×**) | 9,275 (**1.98×**) | 6,342 (**2.90×**) | 6,228 (**2.95×**) |

**Decode — ms per generated token (cache):**

| T | teacher B1 | hybrids B1 (R1 / R2) | all-linear B1 (R3 / R4) | teacher B8 | hybrids B8 | all-linear B8 |
|---|---|---|---|---|---|---|
| 2,048 | 31.0 (148 MiB) | 46.8 / 44.7 (79 MiB) | 51.7 / 49.5 (57 MiB) | 29.6 (1.1 GB) | 48.0 / 45.8 (0.6 GB) | 54.4 / 51.5 (381 MiB) |
| 32,768 | 31.0 (2.3 GB) | 46.4 / 44.6 (0.6 GB) | 51.9 / 49.3 (57 MiB) | 56.3 (18.4 GB) | 48.2 / 45.8 (4.9 GB) | 54.0 / 51.6 (381 MiB) |
| 65,536 | 34.2 (4.6 GB) | 47.6 / 45.3 (1.2 GB) | 51.8 / 49.4 (57 MiB) | 97.6 (36.9 GB) | 48.2 / 45.7 (9.5 GB) | 54.1 / 51.2 (381 MiB) |
| 131,072 | 41.5 (9.2 GB) | 49.0 / 47.1 (2.3 GB) | 51.3 / 49.5 (57 MiB) | **OOM** (needs ~74 GB) | 56.1 / 56.0 (18.7 GB) | 54.2 / 51.1 (381 MiB) |

**Decode, batch 32 and 64 — ms per generated token (cache):**

| T | teacher B32 | hybrids B32 (R1 / R2) | all-linear B32 (R3 / R4) | teacher B64 | hybrids B64 | all-linear B64 |
|---|---|---|---|---|---|---|
| 2,048 | 30.2 (4.6 GB) | 47.6 / 47.4 (2.3 GB) | 53.9 / 52.1 (1.5 GB) | 39.3 (9.2 GB) | 47.0 / 45.9 (4.6 GB) | 53.8 / 51.6 (3.1 GB) |
| 4,096 | 36.1 (9.2 GB) | 48.3 / 46.6 (3.5 GB) | 53.5 / 52.0 | 63.1 (18.5 GB) | 47.9 / 46.3 (6.9 GB) | 53.8 / 51.9 |
| 8,192 | 56.9 (18.4 GB) | 47.4 / 46.4 (5.8 GB) | 54.1 / 51.8 | 110.7 (36.9 GB) | 47.2 / 45.7 (11.5 GB) | 53.5 / 51.6 |
| 16,384 | 98.2 (36.9 GB) | 47.1 / 45.9 (10.4 GB) | 53.6 / 51.2 | OOM | 66.2 / 65.9 (20.7 GB) | 53.6 / 51.8 |
| 32,768 | OOM | 57.4 / 57.2 (19.6 GB) | 53.6 / 51.5 | OOM | 113.7 / 113.5 (39.2 GB) | 53.6 / 51.8 |
| 65,536 | OOM | 98.9 / 98.6 (38.0 GB) | 53.5 / 51.3 | OOM | OOM | 53.5 / 51.7 |
| 131,072 | OOM | OOM | 53.8 / 52.0 (1.5 GB) | OOM | OOM | 53.6 / 51.3 (3.1 GB) |

Every OOM is now a real one (the cache alone would exceed the GPU: teacher ≥ 74 GB, hybrids ≥ 78 GB). History: the first batch-32/64 run (debug, 2026-10-26) used the v1.1 chunking (4,096 tokens per row per prefill call) and ran out of memory on the MLP activations (failed allocations 4.5 / 9 GiB = B × 4,096 × 18,432 × 2 bytes); those points were deleted (backup `_v11_oom/`) and re-measured with v1.2 (1,024 / 512 tokens per row; jobs 62334075 + chain). All batch-32/64 jobs ran on exclusive debug nodes (`--exclusive` in the sbatch; empty GPU process list in every log); their short-context times reproduce the lprod batch-1/8 run (e.g. R3 53.5–54.4 ms at every batch). Single slow steps (R1/R2 B64 × 32k: max 848 / 884 ms, allocator near full memory) do not affect the median.

Full tables and figures: `plot_speed.py` → `speed_table.md`, `figs/prefill_latency|prefill_speedup|prefill_memory|decode_latency|decode_cache`.

| finding | evidence | status |
|---|---|---|
| ⭐ **Prefill: the students overtake the teacher between 8k and 16k tokens; at 128k hybrids are 2.0× and all-linear 2.9× faster** | crossover ≈ 10k (GDN) / 12k (KDA); log-log slope 64k → 128k: teacher 1.67, hybrids 1.29, all-linear 1.00 (= linear in T). Throughput: teacher 21.5k tok/s at 2k → 7.1k at 128k; all-linear constant ≈ 21k; hybrids ≈ 14k at 128k | ✅ matches Kimi Linear's picture (comparable at 4–16k, gains at ≥ 128k) |
| Below ~8k the students are slower (0.33–0.49× at 128 tokens) | fla's chunked kernels have a fixed per-call cost (≈ 60–90 ms at T = 128 vs 31 ms) | ✅ a constant overhead, irrelevant at long context |
| ⭐ **Decode memory is the large win** | KV cache at 128k: teacher 9.2 GB, hybrids 2.3 GB (**3.9×** smaller), all-linear **57 MiB constant** (**163×**). Cache break-even ≈ 800 tokens (B9a predicted 670; measured states ~20 % above the tensor sum — allocator granularity). Batch 8 × 64k: teacher 36.9 GB, hybrids 9.5 GB, all-linear 381 MiB | ✅ |
| **Batch-1 decode: the teacher is faster at every length up to 128k** (31–42 vs 44–52 ms/token) | HF eager decode is launch-bound (~6 ms of weight reads per step on an A100); a KDA/GDN layer launches more small kernels than a FlashAttention layer, and the students have 8–13 % more parameters. The teacher's step time rises only from 64k (34 → 42 ms), the students' stays flat | ✅ a framework-overhead result; with CUDA graphs / fused kernels the absolute numbers would shrink — the flat-vs-rising shape is the robust finding |
| ⭐ **Batch-8 decode: the students are faster from ~32k, 1.8–2.1× at 64k, and the teacher runs out of memory at 128k** | teacher 29.6 → 35.6 (16k) → 56.3 (32k) → 97.6 ms (64k) → OOM; students flat 45.7–54.4 ms (hybrids 56 at 128k). Throughput at 64k: teacher 82 tok/s, students 148–175 | ✅ the serving advantage appears with batch × context, as expected for memory-bound attention |
| ⭐ **Decode step time is set by the bytes of cache read per step — the same curve for teacher and hybrids** | teacher ≈ 30 ms until ~9 GB of cache, then rising: 18.4–18.5 GB → 56–63 ms, 36.9 GB → 97.6–110.7 ms (B8 × 64k, B32 × 16k, B64 × 8k). Hybrids at the same cache size give the same time: 19.6–20.7 GB → 57–66 ms, 38–39 GB → 98.6–113.7 ms — but that is 4× more tokens (2.1M vs 524k), because 9 of 36 layers keep a KV cache. All-linear: 1.5–3.1 GB at any length → flat 51–54 ms up to B64 × 128k = 8.4M tokens. Effective cache bandwidth ≈ 0.4 TB/s, well below the A100 peak (probably because HF `DynamicCache` appends with `torch.cat`, rewriting the cache every step) | ✅ so the decode crossover moves to shorter contexts as the batch grows: none up to 128k at B1, ~16–32k at B8, ~4–8k at B32, ~2–4k at B64. **Capacity on one 64 GB A100:** teacher ≈ 0.5M cached tokens, hybrids ≈ 2.1M (4×), all-linear not limited by the cache (8.4M tested, 3 GB) |
| **Batching is free for the students in this implementation** | per-step time is the same at batch 1 and 64 (hybrids 45–48, all-linear 51–54 ms), so throughput grows ~linearly: all-linear B64 **1,190–1,250 tok/s at any context up to 128k**; the teacher reaches 2,140 tok/s at B64 × ≤ 1k but cannot run B8 × 128k at all | ✅ |
| Prefill peak memory is the same for all five (≈ 16.4 GB above weights at 128k) | dominated by the MLP activations (identical MLPs), not attention (FlashAttention-2 has no T² buffer) | ✅ the conversion buys nothing in prefill memory |
| GDN is slightly faster than KDA | prefill ≤ 2 %, decode 1–2.5 ms/token (3–5 %) in both layouts and at every batch size; identical cache size |  ✅ small compared with KDA's quality lead (arithmetic +5.5 to +9.5, KL −8 to −15 %) |
| Quality at long context not measured | all students trained on ≤ 2,048-token sequences; speeds on random tokens; the teacher's 128k point lies beyond its configured 64k context (valid for timing, not for quality) | ⬜ limitation: speed at 128k says nothing about quality at 128k |

## 9. Methodological findings worth reporting

- Held-out CE/KL scripts (`test_r1_ce.py` → `ce_outliers`, `heldout_kl.py`) use DistillKit's own definitions (checked against `kl.py` / `cross_entropy.py` / `trainer.py`, 2026-10-19): KL(teacher ‖ student) on fp32 log-softmax at T = 2, summed over the vocabulary, × T², averaged over tokens; CE = the HF causal-LM loss (shifted, token mean). Differences, all negligible or checked: the last position is dropped (2047 of 2048 tokens); per-sequence means are averaged (docs layout: documents weighted equally); logits are not cut to the tokenizer's vocabulary as DistillKit does. Cross-check: R1-S2-math maths KL in training format 0.3045 vs DistillKit's own final `eval_loss` 0.303 (different 200 documents). ✅
- A held-out sequence that does not start with BOS can make some students look catastrophically broken (R5, R1-S1-math: up to +7 nat on single sequences) while they are normal on training-format input; build evaluation sequences the way training builds them, or at least start each with BOS. ✅

- Packing without document isolation is not the dominant term in the measured gap: removing most boundaries (`MIN_DOC_TOK=600`) left the web gap unchanged to 3 parts in 10,000. ✅
- A 600-second NCCL barrier around rank-0 preprocessing kills multi-node runs once the corpus grows (R5 missed by 1 %); fixed by warming the dataset cache in a single process. ✅
- Mean metrics hid R5's damage; only per-sequence distributions showed it. ✅
- Two draws from the same held-out split flipped the sign of the layout effect — small effects need paired designs. ✅
- 0.45 % of held-out texts also occur in stage-1 training (exact duplicates in FineWeb-Edu). ✅