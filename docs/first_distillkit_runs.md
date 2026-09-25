# First DistillKit Runs — Stage 1 with the KDA Student

*The live document. `claude/startup_plan_and_pitfalls.md` = infrastructure (closed). `claude/building_the_student.md` = the model, Tracks A/B/C. `claude/track_b.md` = generation, closed.*

*Revision 20 (2026-09-25) — ⭐⭐ **MMLU and GSM8K full runs landed, and MMLU is the most important number in this document.** Teacher **0.6500**, student **0.3055** — **13.9 % relative**, against the reference hybrid's 79.0 % and against the **55–65 % band this document pre-registered**. GSM8K full: **0.0243 strict** against a teacher at **0.5353** — 4.5 % relative, and **not zero** (5.8 σ, ~32 of 1,319). ⭐⭐ **The pattern is bimodal and the dividing line is the SHOT COUNT, not the scoring mode:** three 0-shot benchmarks hold at 79–89 %, both 5-shot benchmarks collapse. **That falsifies revision 18's "multiplicative error along a generated chain" as the general account** — MMLU is one forward pass and an argmax over four options. Two mechanisms remain confounded (few-shot ICL vs prompt-format distance) and **ARC-C at 5-shot separates them in one job**. Consequence: two of five quality axes are at the floor and cannot discriminate KDA from GDN, which doubles the weight on open decision 2.*

*Revision 19 (2026-09-23) — ⭐⭐ **the control ran, and it overturned revision 18's reading.** A third corpus — Tulu 3, instruction and code, first document Terraform — shows a CE gap of **+0.3320, i.e. 2.08× the web-text gap**, statistically indistinguishable from mathematics' 2.11×. **Mathematics is NOT special; distance from the training distribution is.** Revision 18 called the damage "localised to mathematical text" and leaned that toward the frozen-MLP explanation; that did not survive. ⭐ Tulu 3's own maths content cannot explain it — the gap mixes linearly, so reaching 2.08× from a 1.0×/2.11× blend would require ~97 % maths, and Tulu 3 is nowhere near that. **Consequence for open decision 2: the corpus moves from "unruled-out alternative" to "best-supported explanation", and unfreezing the MLPs on FineWeb-Edu alone may not be the right lever.***

*Revision 18 (2026-09-22) — **what stage 1 did NOT recover.** Track B closed, so the generative benchmarks finally ran: **GSM8K reads ~0 against a teacher at ~0.52–0.60**, where the reference's hybrid keeps 36.5. It is **not a generation bug** — the identical prompts with `use_cache=False` produce **byte-identical** text. And it is **not uniform degradation**: the CE gap is **+0.1595 on web text and +0.3372 on mathematical text**, ~19 σ apart, rising to **2.82×** when short documents are filtered out, so packed-document boundary density is ruled out too. ⚠️ Its attribution of that gap to mathematics specifically is **superseded by revision 19**.*

*Revision 17 (2026-09-20) — **the blog post's token budget is now known, and the benchmark comparison is placed.** Their evaluated students had **300 M stage 1 + ~1 B at 32k with MLPs unfrozen**; ours is stage 1 only. Matching their teacher row shows their chart uses ARC-C **plain acc** and HellaSwag **acc_norm** — raw, that is **47.78 vs their 48.5** on ARC-C and **72.40 vs their 74.3** on HellaSwag, with the same teacher on both sides — on 23 % of the tokens. ⚠️ Relative-score figures quoted anywhere in this document are **computed here from those raw numbers**, not published by Arcee. ⭐ And their own numbers show **KDA-only beating the hybrid on MMLU and ARC-C**, which together with B9a's memory figure makes `kda_only` the most valuable second run.*

*Revision 16 (2026-09-20) — **held-out CE measured: the student closed 97.2 % of the initialization gap.** 2.4559 against the teacher's 2.2963 on 131 k held-out tokens, perplexity 11.7 vs 9.9. A5b's 2.3030 and 7.9412 both reproduce, so the axis is sound. **Identical numbers under transformers 5.8.1 and 5.9.0.** Also: where the 300 M token count is confirmed in the log, and why `save_total_limit: 1` is the wrong setting for the next run.*

*Revision 15 (2026-09-20) — **the R1 student is benchmarked.** ARC-C 0.5205, HellaSwag 0.7240, WinoGrande 0.6630 against a teacher at 0.5922 / 0.7816 / 0.7048 — **79%, 89% and 80% of the teacher's above-chance score, after stage 1 alone.** Also settles the transformers-version question: **do not downgrade to match the reference.***

*Revision 14 (2026-09-20) — **corrects revision 13's account of the KL gap.** Reading `blogpost_yaml.txt` line by line kills two of three reasons. Same teacher, same loss weights, **same KL temperature**, same LR, same schedule shape and floor. **One clear difference remains: the corpus.***

*Revision 13 (2026-09-19) — **R1 finished, and it worked.** 4,580 steps, 300 M tokens, 3.27 h, no crash. **hs_cosine 0.841 → 0.070, KL 10.13 → 0.35, CE 7.918 → 2.26.** Also: the **teacher baseline** for three of the four benchmarks, and a new trap — the sbatch's provenance line records the *repo* hash, not the code the model actually ran.*

*Revision 12 — restructured; A9 closed document isolation in favour of path (a).*

---

## 🧭 How to read this document

| Part | What's in it | When you need it |
|---|---|---|
| **1. Now** | status, what R1 produced, what it did NOT produce, open decisions | every day |
| **2. The runs** | run log, results, what R0 and R1 each established | after each job |
| **3. Reference** | config settings and why, pre-flight, budget arithmetic | when writing a config |
| **4. Closed** | document isolation — the whole story, settled | when writing the thesis |
| **5. Traps** | the accumulated list | before every run |

---

# PART 1 — NOW

## Status

| | |
|---|---|
| Students | ✅ four, rebuilt from `d7cd0a0`, build verified deterministic |
| Pipeline | ✅ R0 clean, **R1 clean** |
| Document isolation | 🔵 closed — path (a), see Part 4 |
| **R1 — stage 1, 300 M tokens** | ✅ **COMPLETE, job 58141472** |
| Teacher baseline | ✅ **all five** — ARC-C / HellaSwag / WinoGrande / MMLU / GSM8K |
| **Student benchmarks** | ✅ **all five** + held-out CE — and **MMLU is the headline**, see below |
| **Generation** | ✅ **Track B closed** — `claude/track_b.md` |
| **GSM8K / IFEval** | ✅ GSM8K full (1,319) · ⬜ IFEval still to run |
| B9b decode throughput | ⬜ the speed half of the architectural claim |
| GDN sibling student | ⬜ Track C — unblocked, code written |

## ⭐ R1 worked

| Signal | At init | After 300 M tokens | Target |
|---|---|---|---|
| **`hs_cosine`** ⭐ | 0.841 | **0.070** | *falling* — the pre-registered pass condition |
| **KL vs teacher** | 10.13 | **~0.35** | Arcee report ≈ 0.2 at this scale |
| **Cross-entropy** | 7.918 | **~2.26** | teacher's own held-out CE is 2.3030 |
| Held-out combined loss | — | **0.5686** | monotone down at all 9 eval points |
| `grad_norm` | 6.36 @ step 10 | 0.057 | no spike, no NaN |

⭐ **`hs_cosine` was the number the doc pre-committed to before the run**, precisely because R0's 20 steps had shown it *rising*. It fell by 92%. Writing the pass condition down in advance is what makes this a result rather than a post-hoc reading.

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

**So the "different teacher–student pairing" reason was simply false**, and "a schedule that hit its floor" is not a difference either. What survives:

1. ⭐ **The corpus.** KL measures agreement with the teacher *on the data you show it*. FineWeb-Edu is raw web text; their mix is curated instruction data. **This is a mechanism, not a measurement** — the honest form for the thesis. ⭐ **And it is the same single difference that the GSM8K finding below cannot rule out**, which makes it the most load-bearing uncontrolled variable in the project.
2. **Schedule length is not comparable.** `num_train_epochs: 1` stretches their cosine over one epoch of a corpus whose size they never state, so their "KL ≈ 0.2 at ~300 M tokens" may be a point *passed through* mid-schedule rather than an endpoint. Ours is an endpoint by construction.
3. **Their global batch is unknown** and, at `per_device 4`, likely larger than our 32.

**How to write it:** ours reached token-averaged KL ≈ 0.35 at 300 M tokens at temperature 2.0, against ≈ 0.2 reported by Arcee at the same token count, under an otherwise identical recipe, on a different and less curated corpus.

### ⭐ Held-out cross-entropy — the single cleanest result

`test_r1_ce.py`, 64 × 2048 = 131,072 held-out tokens, sdpa, bf16:

| model | CE (first 8) | CE (all 64) | ± se | perplexity | vs teacher |
|---|---|---|---|---|---|
| **Teacher** AFM-4.5B-Base | **2.3030** | 2.2963 | 0.0400 | **9.9** | — |
| hybrid31, **untrained** | 7.9274 | 7.9398 | 0.0195 | 2,806.8 | +5.6435 |
| **hybrid31, R1 @ 300 M** | 2.4383 | **2.4559** | 0.0392 | **11.7** | **+0.1595** |

⭐ **The student closed 5.484 of the 5.644-nat gap — 97.2 % — in 300 M tokens.** Perplexity 11.7 against the teacher's 9.9, i.e. **17.3 % higher** (`exp(0.1595) = 1.173`). That single sentence is the most compact statement of what stage 1 achieved.

**The first-8 column is the validity check, and it passes.** The teacher reads **2.3030**, reproducing A5b's published number exactly. The untrained hybrid reads 7.9274 against A5b's 7.9412 and R0's 7.918: three measurements spanning 0.023 nats.

📌 **That small spread is itself informative.** The ★ parameters are random, so **"the untrained hybrid" is a distribution, not a point**, and ~0.02 nats is the scale of that seed noise. Any claimed effect smaller than that is noise.

⚠️ **The printed standard errors are the wrong error bar for the comparison.** ±0.040 is the spread *across sequences*, dominated by how hard each passage is — and both models see the same passages. **Quote the PAIRED per-sequence difference**, whose standard error is ~0.007, six times smaller. `test_r1_ce_math.py` computes it correctly; `test_r1_ce.py` does not.

⭐ **Cross-environment check: identical numbers under transformers 5.8.1 and 5.9.0.** The 5.8.x/5.9.x skew does not move this model's logits. *(Independently confirmed later: `DynamicCache.get_mask_sizes` is byte-identical between the two versions.)*

---

## ⭐⭐ What stage 1 did NOT recover — GSM8K, and where the damage sits

*Track B closed on 2026-09-22, which made the generative benchmarks runnable for the first time. The first thing they produced was a problem.*

### The number

GSM8K, 5-shot, smoke on the debug queue:

| | n | GSM8K 5-shot |
|---|---|---|
| **teacher** | 20 | **0.60 ± 0.11** — consistent with Arcee's published **0.52**, so the harness is sound |
| **student, cache ON** | 20 | 0.00 strict / 0.05 flexible |
| **student, cache OFF** | **5** | 0.00 / 0.00, and **byte-identical generated text** on the five it shares |

⚠️ **The cache-OFF run used `LIMIT=5`, not 20.** It is a control, not a score: it establishes that the first five generations are character-for-character identical with and without the cache. Do not read its 0.00/0.00 as a measurement, and do not compare its score to the 20-item rows.

⚠️ **`--limit` is a pipeline test, not a measurement.** 0/20 is consistent with a true rate up to ~14 %. The full 1,319-item runs are queued; **do not quote 0.00 until they land.**

Against the reference, whose hybrid scores **36.5** where its teacher scores 52.1 — 70.1 % relative — this is the one benchmark where we are not close.

### It is not a generation bug

⭐⭐ Running the identical prompts through the full-recompute oracle (`use_cache=False`) produced **the same characters**, to the truncation point, including the same `#### 150,000` and the same `[invalid]` extractions. Not "agreeing within noise" — identical, despite the cached path switching Triton kernels mid-generation. **Track B is exonerated by its own oracle.**

What the samples show: the student reproduces the few-shot **form** perfectly — `<<16*7=112>>` annotations, the `####` marker, clean termination on one of three — and fails the **arithmetic**: `2 + 0.5 = 2`, `150,000 − 120,000 = 150,000`.

### ⭐ And it is not uniform degradation — the damage is LOCALISED

`test_r1_ce_math.py` puts the student-vs-teacher CE gap on mathematical text beside the same gap on web text, in one process, same dtype, same kernels, using the **paired** per-sequence statistic.

**Unfiltered** — the corpora as they come:

| corpus | docs/seq | teacher | student | gap | paired se | ppl ratio |
|---|---|---|---|---|---|---|
| web (FineWeb-Edu held-out) | 2.3 | 2.2963 | 2.4559 | **+0.1595** | 0.0070 | 1.173× |
| math (MetaMathQA) | 9.5 | 0.8673 | 1.2045 | **+0.3372** | 0.0064 | 1.401× |

**gap_math / gap_web = 2.11×.** The difference between the gaps is 0.178 against a combined se of ~0.0095 — about **19 σ**. The uniform-degradation reading is dead.

**`MIN_DOC_TOK=600`** — the boundary-density control, since MetaMathQA averaged 216 tokens/doc against FineWeb-Edu's 910 and packing without isolation means every boundary leaks stale context:

| corpus | docs/seq | teacher | student | gap | paired se | ppl ratio |
|---|---|---|---|---|---|---|
| web (FineWeb-Edu held-out) | **1.1** | 2.3927 | 2.5519 | **+0.1592** | 0.0070 | 1.173× |
| math (MetaMathQA) | **2.6** | 0.5430 | 0.9914 | **+0.4484** | 0.0127 | 1.566× |

⭐⭐ **The ratio went UP, to 2.82×.** Boundary density was the hypothesis; removing boundaries should have shrunk the effect, and it grew by a third. **The confound is ruled out, and the finding survives it.**

⭐ **The web gap is invariant and the math gap is not.** Filtering changed *which* documents were sampled — the teacher's absolute CE moved from 2.2963 to 2.3927 on web and from 0.8673 to 0.5430 on math — yet the **web gap held at +0.1592 against +0.1595**, three parts in ten thousand. Over the same change the math gap grew 33 %. That is a strong validation of the paired-gap methodology *and* a real finding underneath it: **longer context widens the gap off-distribution and not on it**, which is the failure mode a fixed-size recurrent state is expected to have — its memory of what came before is lossy, and the loss compounds with distance.

### ⭐⭐ The control — and it overturns the reading above

A third corpus, **Tulu 3** (instruction and code; the first document is Terraform HCL), run in the same process as the other two:

| corpus | docs/seq | teacher | student | gap | paired se | **relative to web** |
|---|---|---|---|---|---|---|
| **web** (FineWeb-Edu — *the training distribution*) | 2.2 | 2.2963 | 2.4559 | +0.1595 | 0.0070 | **1.00×** |
| math (MetaMathQA) | 9.5 | 0.8673 | 1.2045 | +0.3372 | 0.0064 | **2.11×** |
| **instr/code (Tulu 3)** | 5.8 | 1.7218 | 2.0538 | **+0.3320** | 0.0104 | **2.08×** |

⭐⭐ **Mathematics is not special. Distance from the training distribution is.** Two corpora with nothing in common except being off FineWeb-Edu — one of them largely code and instruction-following — show the same doubling, 2.11× and 2.08×, within each other's error bars.

**Tulu 3's own maths content cannot explain this.** The gap is a token-weighted mean, so it mixes linearly: a blend of a 1.00× component and a 2.11× component reaches 2.08× only at `(2.08 − 1.00) / (2.11 − 1.00) ≈ 97 %` maths. Tulu 3 is an instruction mix in which mathematics is a minority. **The contamination that made this a conservative control is not doing the work.**

⚠️ **This supersedes the "localised to mathematics" reading.** The honest statement is now: *the student reproduces its teacher roughly twice as poorly off its training distribution as on it, and GSM8K sits off that distribution while also being multiplicatively sensitive to per-token error.* The maths-specific framing was one corpus short of being tested, and the test did not support it.

⚠️ **One limitation that no control here can remove.** Any student distilled on corpus X will match its teacher better on X than off it — that is generalisation, not damage. Establishing that **2×** is attributable to the *architecture* would need the same recipe run on a non-converted student, which we do not have and cannot cheaply build. **Report the ratio as measured; do not attribute it to KDA without that control.**

📌 **The direction is robust to normalisation; the magnitude is not.** Raw nats 2.11×/2.82×; perplexity ratio 1.401 against 1.173; gap as a fraction of the teacher's own CE 38.9 % against 6.9 %. The corpora have very different entropy (teacher perplexity 9.94 against 2.38). **Quote one, name it, say the sign survives the others.**

### ⚠️ What this does and does not establish

| claim | status |
|---|---|
| The GSM8K collapse is a generation/cache bug | ❌ **ruled out** — byte-identical uncached output |
| The degradation is uniform, and GSM8K only looks bad because exact-match over a chain amplifies it | ❌ **ruled out** — 2.11× at 19 σ |
| It is an artefact of packed-document boundary density | ❌ **ruled out** — the ratio *grows* when boundaries are removed |
| The degradation is concentrated on mathematical text **specifically** | ❌ **ruled out** — Tulu 3 (instruction/code) shows 2.08× against maths' 2.11× |
| The degradation tracks **distance from the training distribution** | ✅ **measured** — both off-distribution corpora at ~2.1× |
| The cause is the frozen MLPs specifically | ⬜ **not established, and now the weaker of the two** |
| The cause is the corpus (FineWeb-Edu is narrow) | ⭐ **best supported** — it is what a distribution-tracking gap predicts |
| The 2× is attributable to the KDA architecture rather than to ordinary generalisation | ⬜ **not established** — would need the same recipe on a non-converted student |

**The defensible sentence is:** *the student reproduces its teacher about twice as poorly off its training distribution as on it, on two unrelated out-of-distribution corpora; GSM8K sits off that distribution and is multiplicatively sensitive to per-token error.*

⭐ **That is still enough to carry open decision 2 — but it points at a different lever than revision 18 thought.**

⚠️ **And language modelling on mathematical text is not the ability to solve a problem.** A model can predict the tokens of a worked solution it could not have produced. This localises representational damage; it is not a second GSM8K.

### Why it matters, and it is not the absolute score

We are not trying to reproduce Arcee's model. A thesis comparing KDA against GDN against the teacher needs one consistent recipe, not their recipe.

⭐ **It matters because a benchmark that reads ~0 for every student cannot discriminate between them.** If KDA and GDN both score zero on GSM8K, the axis is lost — and it is the axis where linear-attention conversions are most often reported to suffer. That is the concrete cost, and a far better argument than "our number looks bad".

---

## ⭐⭐ The full runs landed — and MMLU reframes everything above

*MMLU jobs 58374833 (teacher) / 58276826 (student); GSM8K jobs 58375001 / 58375025, full 1,319 items. 2026-09-25.*

| benchmark | shots | scoring | teacher | student | chance | **relative** |
|---|---|---|---|---|---|---|
| HellaSwag `acc_norm` | **0** | likelihood | 0.7816 | 0.7240 | 0.25 | **89.2 %** |
| WinoGrande `acc` | **0** | likelihood | 0.7048 | 0.6630 | 0.50 | **79.6 %** |
| ARC-C `acc_norm` | **0** | likelihood | 0.5922 | 0.5205 | 0.25 | **79.1 %** |
| **MMLU `acc`** | **5** | likelihood | **0.6500** | **0.3055** | 0.25 | **13.9 %** |
| **GSM8K** strict | **5** | generation | **0.5353** | **0.0243** | ~0 | **4.5 %** |

⚠️ *Relative scores are `(student − chance) / (teacher − chance)`, **computed here**, not published.*

### ⭐⭐ The pattern is bimodal, and the line that separates it is **the shot count, not the scoring mode**

Three benchmarks at 79–89 %; two at 4–14 %. Revision 18's account of GSM8K — *errors compose multiplicatively along a generated chain* — **cannot explain MMLU at all.** MMLU is one forward pass and an argmax over four options: exactly the machinery ARC-C uses, and ARC-C holds at 79.1 %. The generative/discriminative split is **falsified as the general account**. It survives only as a reason GSM8K is 4.5 % rather than 14 %.

**What the two collapsed benchmarks share is that they are the only 5-shot ones.** A 5-shot prompt asks the model to hold five worked exemplars in context and condition on them — **retrieval over distance**, which is precisely what a fixed-size recurrent state is expected to do worst, and precisely what the `MIN_DOC_TOK=600` control already showed (longer context widens the gap off-distribution and leaves it flat on-distribution).

⚠️ **Two mechanisms are confounded here and this document must not pick one yet:**

| | |
|---|---|
| **(a) few-shot in-context learning is what broke** | conditioning on exemplars held at distance — the architectural reading |
| **(b) prompt-format distance** | a templated 5-shot block is far from raw FineWeb-Edu web text — revision 19's "distance from the training distribution", restated |

Every 5-shot task we run is templated and every 0-shot task is natural prose, so the two cannot be separated from these five numbers.

⭐⭐ **One cheap job separates them: ARC-Challenge at 5-shot.** If a benchmark the student holds at 79.1 % 0-shot collapses toward chance at 5-shot, the fault is the *format*, not the task — and (a)/(b) become the live question. If it holds, the collapse is task-specific and MMLU's breadth is implicated instead.

```bash
# DIAGNOSTIC ONLY -- an ADDITIONAL measurement, not a change to the frozen protocol.
# ARC-C 0-shot stays the number of record; label this one separately, forever.
NUM_FEWSHOT=5  # ... on the existing eval_zeroshot_afm.sbatch task list, ARC-C alone
```

### What else the two runs say

⭐ **GSM8K is not zero.** 0.0243 strict ± 0.0042 on 1,319 items — about **32 problems solved**, **5.8 σ above zero**. The smoke run's 0/20 was consistent with a true rate up to ~14 %; the true rate is 2.4 %. **The doc's own caution was correct and the answer sits at the low end of what it allowed.**

⭐ **And the student loses the output format, not only the arithmetic.** Student flexible 0.0349 against strict 0.0243 — roughly fourteen problems where the right number appeared but the `####` marker did not. The teacher is the other way round (strict 0.5353 ≥ flexible 0.5345): it always emits the marker, and flexible-extract occasionally grabs a wrong trailing number instead. **Format compliance is degrading too**, which is itself a few-shot-conditioning signal.

⭐ **MMLU's damage is not concentrated in any subject area.** Relative score by category: humanities **8.2 %**, other **12.1 %**, STEM **15.6 %**, social sciences **19.6 %**. Uniformly near the floor, so this is not "the student forgot chemistry" — it is the task format.

⚠️ **The MMLU teacher row does NOT validate as tightly as the other three.** Ours reads **65.00** against Arcee's published **63.1** — 1.9 points, where ARC-C (55.63 vs 55.6), HellaSwag (78.16 vs 78.0) and GSM8K (53.53 vs 52.1) all land within 1.4. lm-eval's `mmlu` group `acc` is a **micro-average over all 14,042 items**; a macro-average over the 57 subjects is a different number, and some published tables use it. **Say which you report, once, in the methods section** — and note that the 13.9 % relative figure barely moves either way, because the student is 14 σ above chance and nowhere near the teacher.

### ⚠️⚠️ What this costs the thesis

**Two of five benchmark axes are now at the floor.** If the GDN student also reads ~14 % on MMLU and ~3 % on GSM8K — and there is no reason to expect otherwise — **neither can discriminate between the mechanisms.** The KDA-vs-GDN comparison then rests on three 0-shot benchmarks, held-out CE, and the memory and throughput curves.

⭐ **That is the concrete argument for open decision 2, and it is now twice as strong as it was.** Not "our numbers look bad" but: *the study loses two of its five quality axes, including the one every conversion paper reports as the hardest.*

---

## ⭐⭐ Can we compare to the blog post? Yes — and the comparison is flattering everywhere except maths

**Their training budget, from the post's own words:**

> "…only about 300 million tokens seen, token-averaged KL divergence between student and teacher converged to around 0.2"
>
> "one-phase distillation, plus about a billion tokens at 32k sequence length, with a 1e-5 learning rate"

So their **evaluated** models are **300 M (stage 1, 2048) + ~1 B (32k, `lr 1e-5`, MLPs unfrozen)** ≈ **1.3 B tokens**. Ours is **300 M of stage 1 only, MLPs frozen throughout** — about **23 %** of their budget, and missing the stage in which their MLPs were allowed to adapt.

⭐ **Which metric they report — established by matching the teacher row.** Their chart's teacher reads ARC-Challenge **55.6** and HellaSwag (Norm) **78.0**; we measured our teacher at ARC-C `acc` **55.63** and HellaSwag `acc_norm` **78.16**. So their "ARC-Challenge" is **plain `acc`**, their HellaSwag is **`acc_norm`**, and it is the same teacher.

### The raw numbers — what is published and what we measured

| | teacher | hybrid student | source |
|---|---|---|---|
| **ARC-C `acc`** — theirs | 55.6 | **48.5** | their chart, as printed |
| **ARC-C `acc`** — ours | 55.63 | **47.78** | job 58216691 / our student run |
| **HellaSwag `acc_norm`** — theirs | 78.0 | **74.3** | their chart |
| **HellaSwag `acc_norm`** — ours | 78.16 | **72.40** | ditto |
| **GSM8K 5-shot** — theirs | 52.1 | **36.5** | their chart |
| **GSM8K 5-shot** — ours | **53.53** | **2.43** | job 58375001 / 58375025, n=1,319 |
| MMLU `acc` 5-shot — theirs | 63.1 | 55.1 | their chart |
| **MMLU `acc` 5-shot** — ours | **65.00** | **30.55** | job 58374833 / 58276826 |

**47.78 against 48.5, and 72.40 against 74.3, with the same teacher on both sides — and 0 against 36.5.** Three of the four available comparisons are within two points on 23 % of the tokens; the fourth is the whole distance.

### The derived numbers — ⚠️ computed here, not published anywhere

```
relative score = (student − chance) / (teacher − chance)      chance: 0.25 ARC-C, 0.25 HellaSwag
theirs, ARC-C     = (48.5  − 25) / (55.6  − 25) = 76.8 %
ours,   ARC-C     = (47.78 − 25) / (55.63 − 25) = 74.4 %
theirs, HellaSwag = (74.3  − 25) / (78.0  − 25) = 93.0 %
ours,   HellaSwag = (72.40 − 25) / (78.16 − 25) = 89.2 %
```

| relative score *(computed)* | **ours** (300 M, stage 1 only) | **theirs** (≈1.3 B, + 32k stage) |
|---|---|---|
| ARC-Challenge | **74.4 %** | 76.8 % |
| HellaSwag | **89.2 %** | 93.0 % |
| **MMLU** | **13.9 %** ⭐⭐ | 79.0 % |
| **GSM8K** | **4.5 %** | **70.1 %** |

⭐ **Within 2–4 points of their published hybrid on the discriminative benchmarks, on 23 % of the tokens and without stage 3** — and nowhere at all on maths. The plausible mechanism for the first half is that their extra ~1 B tokens went into **32k long-context** capability, which none of these 2048-length benchmarks tests. The second half is the subject of the section above.

**So the two comparisons have different standing, and the thesis must say which is which:**

| comparison | like-for-like? |
|---|---|
| **KL ≈ 0.35 vs their ≈ 0.2 at 300 M** | ✅ **yes** — same stage, same token count, same temperature, same recipe. Only the corpus differs. |
| **Benchmarks vs their published chart** | ⚠️ **no** — theirs had 4.3× the tokens and a long-context stage with MLPs unfrozen. Ours is the *stage-1-only* point on their curve, which they never published. |

📌 **That second row is an opportunity, not a weakness.** "What does stage 1 alone buy?" is a question the reference leaves open, and our numbers answer it — including the uncomfortable part.

### ⭐ Their own numbers argue for training KDA-only

Raw, as printed, then the same relative-score transformation ⚠️ *(computed here)*:

| | MMLU | ARC-C | HellaSwag | GSM8K |
|---|---|---|---|---|
| teacher *(published)* | 63.1 | 55.6 | 78.0 | 52.1 |
| hybrid 3:1 *(published)* | 55.1 | 48.5 | 74.3 | 36.5 |
| kda_only *(published)* | 55.8 | 49.9 | 74.3 | 26.8 |
| FLP *(published)* | 44.0 | 42.4 | 62.6 | 21.0 |
| — hybrid, relative *(computed)* | 79.0 % | 76.8 % | 93.0 % | **70.1 %** |
| — **kda_only, relative** *(computed)* | **80.8 %** | **81.4 %** | 93.0 % | 51.4 % |
| — FLP, relative *(computed)* | 49.9 % | 56.9 % | 70.9 % | 40.3 % |

**KDA-only beats the hybrid on MMLU and ARC-C in Arcee's own published numbers**, ties on HellaSwag, and loses only on GSM8K — the one task needing multi-step retrieval over a long chain. Put that beside B9a, where **kda_only is 48.9× smaller at 32k against the hybrid's 3.77×**, and KDA-only is better on *both* axes for everything except maths.

📌 **And note the pattern repeats at the mechanism level**: whatever a fixed recurrent state costs, it costs it on maths first. Their kda_only drops from 70.1 % to 51.4 % on GSM8K while holding everywhere else — the same shape as our own measurement, in their numbers.

⚠️ **That makes `afm_kda_only_init` the most valuable second training run**, ahead of the de-novo and FLP ablations.

### ⭐ The student benchmarks — job of 2026-09-20, `checkpoint-4580`

| Benchmark | Teacher | **R1 student** | Gap | **Relative score** |
|---|---|---|---|---|
| ARC-Challenge, `acc_norm` | 0.5922 | **0.5205** | −7.2 pts | **79.1 %** |
| HellaSwag, `acc_norm` | 0.7816 | **0.7240** | −5.8 pts | **89.2 %** |
| WinoGrande, `acc` | 0.7048 | **0.6630** | −4.2 pts | **79.6 %** |
| **MMLU, `acc` (5-shot)** | **0.6500** ±0.0037 | **0.3055** ±0.0039 | **−34.5 pts** | **13.9 %** ⭐⭐ |
| **GSM8K, 5-shot** strict | **0.5353** ±0.0137 | **0.0243** ±0.0042 | −51.1 pts | **4.5 %** |
| *(GSM8K flexible-extract)* | 0.5345 | 0.0349 | | 6.5 % |

⭐ **What this is.** Twenty-seven of thirty-six token-mixing layers replaced by a recurrent linear-attention mechanism, initialized from noise plus a projection transplant, trained for **300 M tokens of stage 1 alone** — no stage 2, no long-context stage, MLPs and embeddings frozen the whole way — and the model retains **79–89 %** of its teacher's above-chance ability on three standard benchmarks **while losing mathematics entirely**. Both halves of that sentence are the result.

⭐⭐ **That warning was right, and it under-called it.** The pre-registered band was 55–65 % relative on MMLU. **It came in at 13.9 %** — see the section below, which is now the most important result in this document.

📌 **The three numbers are not equally informative.** HellaSwag at 89 % is the easiest of the three for a linear-attention model. WinoGrande's 79.6 % rests on a 0.50 chance floor, so its relative score is the most sensitive to noise (±0.013 raw becomes ±6 points relative). ARC-C at 79 % is the most demanding.

⚠️ **The evaluation ran against `output/r1_stage1_hybrid31/checkpoint-4580`, not the top-level directory.** The weights are the same model; only the `.py` copies differ. **Pick one of the two paths and use it for every student number.**

⚠️ **Pass `use_cache=True` explicitly in `model_args`, for teacher and student alike.** The Trainer writes `use_cache: false` into every checkpoint, and **Arcee ship their published configs the same way** — so a flag flipped on one side only makes the student O(n) and the teacher O(n²). Verified: `AFM-4.5B-Base` loads with `config.use_cache = False`.

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

**The LR schedule did exactly what the config said.** Peak 1e-3 after ~114 warmup steps, cosine decay, floor **1e-5** = `min_lr_rate: 0.01 × 1e-3`, confirming the DeepSpeed workaround end to end.

**An arithmetic check worth doing once.** `0.2 × 2.2626 + 0.2 × 0.3526 + 0.6 × 0.0705 = 0.5653` against a reported 0.5621. ✅ **The loss weights are the ones in the YAML.**

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

**Held-out, at each of the nine evals:** 0.7429 → 0.6954 → 0.6549 → 0.6308 → 0.6109 → 0.5923 → 0.5806 → 0.5721 → 0.5688 → 0.5686. **Monotone throughout, no overfitting anywhere.**

⚠️ **Read the flattening at the end correctly.** The last 500 steps moved held-out loss by 0.0002 — but the learning rate had been on its 1e-5 floor for roughly the final 600 steps. **That is the schedule ending, not the model saturating.**

**The consequence:** extending this run is **not** a `resume_from_checkpoint`. A cosine schedule that has annealed to its floor cannot be continued. More tokens means **a new run with `max_steps` set to the larger budget from the start.**

## Open decisions — for Anamaria

> **1. Token budget.** 300 M reached KL ≈ 0.35 against Arcee's reported ≈ 0.2, with the held-out curve still falling until the LR floor. 1.3 B is ~12.5 h at the measured 25,455 tok/s and ~2,025 local hours against ~108,000 remaining. Given the curve, is a longer stage 1 the best use of that budget, or are the ablations worth more?
>
> **2. Scope — and MMLU has made this the decision that matters.** Stage 1 alone reaches **ARC-C 47.8 and HellaSwag 72.4 — within 1–2 points of the reference's finished model** — but **MMLU reads 13.9 % relative against the reference's 79.0 %, and GSM8K 4.5 % against their 70.1 %**. ⭐ **Both collapsed benchmarks are the 5-shot ones**; all three that held are 0-shot. The measured cause is that the student reproduces the teacher **~2× worse off its training distribution**, on maths *and* on instruction/code alike. Two levers follow, and they are not the same:
>   - **(a) unfreeze the MLPs** (RADLADS stage 2 / Arcee stage 3) — about one extra stage-1-sized run at 2048;
>   - **(b) broaden the corpus** — distillation only teaches the student to match the teacher *on the data it sees*, and FineWeb-Edu is raw web text where Arcee used a curated instruction mix. `tulu3` and `dolci` are already on disk.
>
>   ⚠️ **(b) is now the better-evidenced of the two**, and the two are independent — a run could do either, or both. Without one of them GSM8K reads zero for *every* student and stops being usable as a comparison axis.
>
> **3. Ablations.** Four students are built (hybrid 3:1, KDA-only, FLP, no-weight-transfer), each ~3.3 h. How many get a full run? ⭐ Arcee's own numbers put **kda_only** ahead of the hybrid on MMLU and ARC-C, and B9a puts it at 48.9× smaller state against 3.77× — so it is the strongest candidate.

⚠️ **If decision 2 is taken, use `lr 1e-5`, not stage 1's `1e-3`.** Arcee's stage 3 does. Unfreezing 4.9 B parameters at the stage-1 rate risks damaging the knowledge that is currently *intact*, and ARC-C 47.8 / HellaSwag 72.4 is the part of this result you least want to break. It is a repair, not a retrain — and the KL term anchors the MLPs to teacher behaviour rather than letting them drift.

⚠️ **And unfreezing may not be sufficient on its own.** Distillation teaches the student to match the teacher *on the data it sees*. The Tulu 3 control says the gap tracks distribution distance, so more capacity to adapt does not obviously help if the data never exercises what is missing. ⭐ **Which is why the cleanest next run varies ONE of the two levers, not both** — varying both gives a better model and no answer.

⚠️ **The corpus deviation is the project's most load-bearing uncontrolled variable.** It is the surviving explanation for KL ≈ 0.35 against Arcee's ≈ 0.2, *and* the best-supported explanation for the GSM8K collapse. Two independent findings pointing at one variable is a reason to name it plainly in the methods section rather than bury it in a deviations table.

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

⚠️ **The two hashes in R1's row are not the same thing, and this is a trap.** The sbatch printed **3fb1104** from `git log -1`; the model that trained is the *checkpoint's* frozen copy, **3aafffc**. Proof from the run's own log: 48 `[AFMKDA-PROBE]` lines appear, and those prints had already been deleted from the repo. **The provenance line records the repo, not the code.**

⭐ **Fixed since.** The provenance block now prints `build_info.json` and `sha1sum $STUDENT/*.py`, and `afm_kda` is committed clean at **`d7cd0a0`** with `logs/` and `__pycache__/` gitignored so `git status --porcelain` is a meaningful signal.

## Results

| Metric | Teacher | At init | **After stage 1 (R1)** | After stage 2 (if run) |
|---|---|---|---|---|
| Held-out CE | **2.2963** | 7.9398 | **2.4559** (+0.1595) | |
| Held-out perplexity | 9.9 | 2,806.8 | **11.7** | |
| Token-averaged KL | 0 | ~10.1 | **~0.35** | |
| `hs_cosine` | 0 | 0.841 | **0.070** | |
| Held-out combined loss | — | — | **0.5686** | |
| **MMLU** (5-shot, acc) | **0.6500** ±0.0037 | — | **0.3055** ±0.0039 | |
| **ARC-Challenge** (0-shot, acc_norm) | **0.5922** ±0.0144 | — | **0.5205** ±0.0146 | |
| **HellaSwag** (0-shot, acc_norm) | **0.7816** ±0.0041 | — | **0.7240** ±0.0045 | |
| **WinoGrande** (0-shot, acc) | **0.7048** ±0.0128 | — | **0.6630** ±0.0133 | |
| *(ARC-C raw acc)* | 0.5563 | — | 0.4778 | |
| *(HellaSwag raw acc)* | 0.5882 | — | 0.5449 | |
| **GSM8K** (5-shot, strict exact_match) | **0.5353** ±0.0137 | — | **0.0243** ±0.0042 | |
| IFEval | ⬜ | — | ⬜ queued | |
| **CE gap on MATH text** | — | — | **+0.3372** (2.11× the web gap) | |
| Parameters | 4.6192 B | 4.9990 B | 4.9990 B | |
| Throughput @ 4 nodes | — | — | **25,455 tok/s** | |

**Teacher baseline, job 58216691**, 6 minutes on the debug queue. Protocol frozen in `claude/eval_zeroshot_afm.sbatch`, `claude/eval_mmlu_afm.sbatch` and `claude/eval_generative_afm.sbatch`: 0-shot for ARC-C, HellaSwag and WinoGrande; 5-shot for MMLU and GSM8K; 0-shot IFEval; no chat template. ✅ **ARC-C, HellaSwag and GSM8K all land on the blog post's published teacher figures**, which validates the harness configuration three independent ways.

## What R1 established

1. **The pipeline scales.** 16 ranks, 4 nodes, ZeRO-2, 4,580 steps, no hang, no rank divergence, one `train_runtime`.
2. **The architecture trains.** 27 randomly-initialized recurrent layers, inside a network frozen everywhere else, learned to reproduce what attention did. This was never guaranteed — it is the project's core hypothesis and it is now evidence.
3. **The freezing worked as designed.** 147 frozen, 477 trainable, on all 16 ranks; `alloc_delta` fell from R0's 12.86 GB to 10.73 as Adam state sharded sixteen ways.
4. **The estimates were conservative in the right direction.** 3.36× scaling against a predicted 3.01×; 3.27 h against 3.7; `epoch` 0.7655 against a hand-computed 0.765.
5. **R0's alarming `hs_cosine` rise was an artifact of run length**, exactly as the doc argued at the time.
6. ⭐ **And what it did not establish**: that the conversion preserves everything. It preserves knowledge and commonsense at 74–89 % relative, and it does not preserve arithmetic at all.

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

⭐ **Check 2 validated all of Track A in one number.** A5b measured 7.9412 offline; R0 measured 7.918 inside DistillKit, on different data, through a saved-and-reloaded checkpoint, with DeepSpeed in the loop. **≈ 2.2 would have meant the teacher was training itself.**

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

**Where 147 comes from.** 624 parameter tensors: 1 `embed_tokens` + 1 `lm_head` + 1 `model.norm` + 72 layernorms + 72 MLP = **147 frozen**; 9 attention layers × 5 + 27 KDA layers × 16 = **477 trainable**. ≈ 945 M of 5.0 B, 19%.

**What the norms are.** AFM is pre-norm; each norm is an RMSNorm with a learned per-channel gain. `input_layernorm` is followed by trainable projections so freezing it is **absorbable**; `post_attention_layernorm` and `model.norm` feed frozen matrices, so those are a genuine constraint, accepted on purpose. **A frozen diagonal scale is free only when what follows is trainable and full-rank.**

⭐ **That last sentence is also the mechanism behind the GSM8K finding.** With the MLPs and the norms feeding them both frozen, the KDA layers must reproduce attention's output *in the MLPs' original basis*, with no slack anywhere to absorb a mismatch. Recall survives an approximate match; exact multi-step arithmetic does not.

### ⚠️ `frozen_res` uses `re.search`, not `re.match`

```python
if any(fre.search(name) for fre in frozen_res):
    param.requires_grad = False
```

So the blog post's bare `norm\.weight` also hits KDA's own `o_norm.weight`. Ours are anchored, and R0 and R1 both printed the predicted 147/477.

### ⚠️ `min_lr` crashes under DeepSpeed — use `min_lr_rate`

```
min_lr_rate = min_lr / optimizer.defaults["lr"]
AttributeError: 'DeepSpeedZeroOptimizer' object has no attribute 'defaults'
```

**`min_lr_rate: 0.01` is exactly `1e-5 / 1e-3`.** ✅ R1's final LR read 1.000e-05. ⚠️ **The coupling is manual** — change `learning_rate` and you must change `min_lr_rate`.

### ⚠️ `prepared_dataset_path` is a trap at scale — leave it out

`save_to_disk` ran at **~60 examples/s** in R0 — 5m33s for 20,000 documents, so **~2 hours** at R1's 500,000, inside the training job. Removing the line skips it; TRL then tokenizes at ~2,700 docs/s with `num_proc: 16`. *(Its key is `sha256({path, seed, num_samples})`, so it also silently hits if a dataset's contents change at the same path. Version dataset directories by name.)*

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

⚠️ **The dataset row is now the project's most load-bearing deviation.** It is the surviving explanation for the KL gap *and* an unruled-out explanation for the GSM8K collapse. Say so once, plainly, in the methods section.

## ⭐ Library versions — the policy, and why

| | transformers | used for |
|---|---|---|
| `distill_env_cu126` | **5.8.1** | building students, R0, R1, all A- and B-series tests |
| `eval-env2` | **5.9.0** | lm-eval-harness, teacher and student benchmarks |
| Arcee's reference code | **4.57** | — never run by us |

### Do NOT downgrade to 4.57 to match the reference

The whole of Track A is built on transformers 5.x and would have to be rewritten: `@strict` dataclass configs and `PreTrainedConfig`, `create_causal_mask`, `GradientCheckpointingLayer`, `ALL_ATTENTION_FUNCTIONS.get_interface`. Downgrading would also re-inherit the six defects the subclassing approach was chosen to avoid.

### Is differing from the reference a problem? **No — and say so explicitly in the thesis**

A model architecture is defined by its config, its weights and its arithmetic, not by a library version. What matters is only that (i) **every model being compared runs under the same version** — teacher and students both do — and (ii) the version is recorded with every number. ⭐ **A3's `0.000e+00` against the unmodified teacher, measured inside 5.x, is the evidence that our implementation is faithful.**

### ✅ The 5.8.1 / 5.9.0 skew — closed by measurement, twice

`test_r1_ce.py` returns identical numbers in both environments, and `DynamicCache.get_mask_sizes` is byte-identical between the two versions. **Record both versions with the numbers and stop worrying about it.** ⚠️ **Do not change `distill_env_cu126`**: it trained R1, and a working training stack is not worth risking for tidiness.

## Pre-flight — things that only bite at scale

- **Pre-warm the remote-code module cache before `srun`.** With `--ntasks-per-node=1` all four local processes share one `SLURM_PROCID`, so a per-rank `HF_MODULES_CACHE` does not help. The single-process `from_pretrained` before `srun` is the fix.
- **~4 minutes of Triton autotuning on step 1 of every job.** Never quote `train_runtime` from a short run.
- **`HF_HUB_OFFLINE=1` is fine**, but DistillKit uses the *teacher's* tokenizer, so that cache must be complete.
- **`eval_strategy: "no"` for any run whose timing you quote.**
- **Commit before launching — and record the *checkpoint's* hash, not the repo's.**

### ⭐ The provenance block that records the code that will actually run

```bash
echo "=== provenance ==="
git -C $WORK_LUKAS/distillkit log -1 --oneline          # the TRAINER: repo hash is correct here
echo "--- afm_kda (the model code) ---"
git -C $WORK_LUKAS/afm_kda log -1 --oneline
git -C $WORK_LUKAS/afm_kda status --porcelain || echo "(clean)"
echo "--- model (the code that will RUN, not the repo) ---"
cat $STUDENT/build_info.json
sha1sum $STUDENT/*.py
echo "=================="
```

⭐ **The `--porcelain` line is the one that earns its keep.** The commit hash only describes what ran if nothing was edited after it. Empty = trustworthy. Non-empty = a half-truth, and it names the files. ⚠️ **This only works because `logs/` and `__pycache__/` are gitignored** — otherwise every test job leaves the tree dirty and the signal is permanently red.

## ⭐ Confirming the token count from the log itself

**Route 1 — from what the trainer measured:**

```
train_samples_per_second 12.43  ×  train_runtime 11,790 s  =  146,550 sequences
146,550 × 2,048 tokens                                     =  300.1 M tokens
```

**Route 2 — from the configuration:**

```
max_steps 4,580  ×  global_batch 32  ×  2,048  =  300,154,880 tokens
```

⭐ **146,550 measured against 146,560 configured** — agreement to four significant figures.

📌 **"Sequences", not "documents".** With `packing: true` every row is 2,048 real tokens, so the token count is exact.

⚠️ **This check belongs in every run's write-up**, because `max_steps` is configuration intent and `train_samples_per_second × train_runtime` is a measurement.

## ⚠️ `save_total_limit: 1` is the wrong setting for the next run

R1 kept one checkpoint, defensible for a cosine schedule that anneals to its floor. **But it throws away the most valuable figure the run could have produced: benchmark quality against training tokens** — precisely the question Anamaria has to decide.

⚠️ **`save_total_limit` keeps the most RECENT N, not a spread.**

```yaml
  save_steps: 1145            # 4580 / 4
  save_total_limit: 4         # -> steps 1145, 2290, 3435, 4580, all kept
```

Four checkpoints ≈ 84 GB with optimizer state. Check `df -h $WORK_LUKAS` before launching.

⭐ **And with Track B closed, a quality-versus-tokens curve can now include GSM8K** — which is the one metric where we do not know whether 300 M was 10 % of what is needed or 1 %.

## Budget arithmetic

| | |
|---|---|
| Train documents | 2,921,000 |
| GPT-2 tokens (FineWeb's own column) | 3.015 B |
| AFM tokens, raw (5,000-doc sample × 970.1) | ~2.84 B |
| **Effective tokens after packing** (774/doc) | **~2.26 B** = one epoch |

At 4 nodes, `tokens/step = 65,536`. **Measured 25,455 tok/s at 4 nodes (R1).**

| Target | max_steps | wall-clock | local hours | share of one epoch |
|---|---|---|---|---|
| **300 M** ✅ done | 4,580 | **3.27 h** (measured) | ~420 | 13% |
| 600 M | 9,155 | ~6.5 h | ~840 | 27% |
| 1.3 B | 19,840 | **~12.5 h** | ~1,620 | 58% |

⚠️ **300 M is Arcee's number.** The larger figures are budget-shaped choices with no convergence evidence behind them.

⚠️ **Truncation, measured on 2,000 documents:** mean 1,073 AFM tokens, median 610, p95 3,293, **10.2% of documents exceed 2048** and **26.6% of tokens are lost to truncation**.

⚠️ **`epoch` is measured against `num_samples`**, not the corpus.

### ⭐ Scheduler facts, verified 2026-09-22

This account may use exactly four QoS: **`boost_qos_dbg`** (30 min, 8 nodes, **priority 80**), **`boost_qos_bprod`** (24 h, but `MinTRES node=65` — **unreachable**, a 4-node job is refused with `QOSMinCpuNotSatisfied`), **`boost_qos_lprod`** (4 days, 8 nodes, priority 40) and **`normal`** (priority 40). **`normal` and `lprod` are the same priority**, so for a 4-node job under 24 h they are equivalent and lprod's only advantage is walltime. Priority is a *weight* in SLURM's scoring, not a queue jump. Full detail in `startup_plan_and_pitfalls.md` Part 4.

---

# PART 4 — CLOSED: document isolation

*The complete story, settled.*

## What the problem is

Rows are 2048 tokens; FineWeb-Edu documents average ~970 AFM tokens. Rather than waste half of every batch on padding, the trainer **packs** — concatenating documents and cutting the stream into fixed rows. A row may hold the tail of A, all of B, and the start of C.

The model doesn't know that. Unless something marks the seams it reads the row as one text. For the nine attention layers that is cross-document attention; for the twenty-seven KDA layers the **recurrent state** flows across the boundary unless reset.

**A quality tax, not a crash.** The reference's published recipe does it. **Your own practical work did too** — `test_config_4B_instruct.yaml` has `packing: true` and no isolation, and produced IFEval 0.6511 loose.

## Where it goes wrong in the code

```python
data_collator=collate_packed_batch if config.dataset.prepacked else None,
processing_class=None   if config.dataset.prepacked else tokenizer,
```

- **`prepacked: false`** (ours). TRL tokenizes, packs and collates; every position is a real token, so **it emits no `attention_mask` at all**.
- **`prepacked: true`.** `collate_packed_batch` stacks whatever columns the dataset has, and `pack_logits.py` writes segment ids into `attention_mask`.

⭐ **So `functionary_packing` and `dataset.prepacked` are two halves of one feature.** We enabled the interpreter and never supplied the data.

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

Under `flash_attention_2`, transformers only builds a mask when something needs masking; a segment mask contains no zeros, so it reads as "all valid" and is discarded.

**So if we packed with segment ids, the student would isolate and the teacher would not** — every token near a boundary would carry a target the student structurally cannot produce. That is worse than no isolation.

⭐ **The reframe that settles it.** As things stand, **student and teacher see exactly the same thing.** The distillation target is self-consistent, and "match the teacher" remains well posed.

## How much does it actually cost?

**Unknown, plausibly small.** ⭐ **And now bounded from one direction by measurement:** `MIN_DOC_TOK=600` cut the packed-document density from 9.5 to 2.6 documents per row on the math corpus and from 2.3 to 1.1 on web — and the student-vs-teacher gap **did not shrink**; on web it was flat to three parts in ten thousand and on maths it grew. **Whatever packing without isolation costs this student, it is not the dominant term in the gap we are measuring.** That is a real, cheap, quotable bound on a question the reference leaves entirely open.

**For the thesis:** one sentence in the methods section — trained with packing and without intra-document masking, as the reference's published recipe does — plus the bound above.

## What the A7 work bought anyway

- **A7** proved `GradientCheckpointingLayer` forwards `**kwargs`, retiring one of the six reference defects.
- **A7b** found that **fla 0.5.2 silently overwrites a supplied `cu_seqlens` whenever a mask is present** — which fired exactly as predicted in Track B, and which GDN turns into a `NameError` instead.
- **A7c** measured the monkey patch at zero calls, and caught `forward` nested inside `__init__`.
- The line of inquiry produced the finding that **batched left-padded generation needs the 2-D mask routed to KDA** — which became half of Track B's fix.

⚠️ **The process lesson.** `main.py`'s two-line use of `functionary_packing` was readable from day one. **Read how a flag is consumed before building on what its name implies.**

---

# PART 5 — TRAPS

- **⚠️ Read how a flag is consumed before building on what its name implies.** `functionary_packing` reads as "enable packing"; it means "assume the data is already packed in functionary's format".
- **⚠️ Confirm a token budget two ways.** R1: 146,560 configured, 146,550 measured.
- **⚠️ `--limit N` is a pipeline test, not a measurement.** 0/20 is consistent with a true rate up to ~14 %, and a generation-time ratio from it is confounded by output length.
- **⚠️ A modest perplexity penalty is not a modest benchmark penalty.** +0.16 nats coexists with 74–89 % relative retention on discriminative tasks and ~0 on a multi-step generative one, because errors compose multiplicatively along a chain.
- **⚠️ A localised result does not name the cause.** "Worse on maths" is equally what the corpus hypothesis and the frozen-MLP hypothesis predict, because the training corpus *is* the web-text distribution being compared against.
- **⭐ The right control can be contaminated and still be worth running, if the contamination is conservative.** Maths inside a supposedly non-maths control biases it toward the maths-like answer — so a web-like result would still be sound.
- **⭐ Compare PAIRED differences, not absolute values, across corpora.** Filtering changed the teacher's absolute CE by 0.10 on web and 0.32 on maths, while the student-teacher gap on web moved by 0.0003. The gap is the stable quantity; the level is not.
- **⭐ The slow oracle keeps earning its keep.** `use_cache=False` was B1's ground truth, and later the one experiment that could exonerate the cache against a bad benchmark number.
- **⚠️ `save_total_limit` keeps the most RECENT N, not a spread.**
- **⚠️ A derived number sitting next to published numbers looks published.** Print the raw numbers and the formula, and label every transformed column.
- **⚠️ Relative score, not raw point gap.** A raw gap flatters tasks with a high chance floor.
- **⚠️⚠️ MMLU falls hardest in every attention-conversion table — and here it fell past the pessimistic band this document pre-registered.** 55–65 % relative was written down in advance; it came in at **13.9 %**. **Do not read three good benchmarks as five.**
- **⭐⭐ A second data point can falsify the explanation you built on the first.** "Errors compose multiplicatively along a generated chain" accounted for GSM8K perfectly and is **powerless against MMLU**, which is one forward pass and an argmax over four options. The property the two collapsed benchmarks actually share is the **shot count**, which nobody was looking at.
- **⚠️ When every instance of a variable co-occurs with another, you have one observation, not five.** All our 5-shot tasks are templated and all our 0-shot tasks are natural prose, so "few-shot ICL broke" and "the prompt format is off-distribution" are indistinguishable until ARC-C is run at 5-shot.
- **⚠️ A harness that validated on three benchmarks has not validated on the fourth.** The teacher matched Arcee's published figures within 1.4 points on ARC-C, HellaSwag and GSM8K, and is **1.9 points off on MMLU** — micro- against macro-averaging over 57 subjects. Check the teacher row on every new task, not once.
- **⚠️ Evaluate one path, consistently.** `output/r1_stage1_hybrid31/` and `.../checkpoint-4580/` hold the same weights but different `.py` copies.
- **⚠️ A setting you fixed on one model is a setting you have now made asymmetric.** Flipping `use_cache` on the student created a confound with the teacher, which ships with `use_cache: false`.
- **⚠️ The remote-code module cache lives under `$HF_HOME`, not `~/.cache/huggingface`.**
- **⚠️ `save_pretrained` DOES copy remote-code `.py` files into a checkpoint.**
- **⚠️ Fix the chain, not the artefacts.** Committing the repo and rebuilding the init students is correctness by construction; copying files into checkpoints is correctness by remembering.
- **⚠️ A signal that always fires is no signal.** An un-ignored `logs/` directory makes the dirty-tree check permanently red.
- **⚠️ Running the old script is silent.** A stale copy produced a full, healthy-looking table identical to the previous run; the tell was a missing print statement, not an error.
- **⚠️ A provenance line can record the wrong thing.** R1's sbatch printed the repo hash; the model that trained was the checkpoint's frozen copy.
- **⚠️ A flat curve at the end of a cosine schedule is the schedule, not the model.**
- **⚠️ An annealed cosine run cannot be extended by resuming.**
- **Check that the reported scalar loss equals the configured weighted sum.**
- **Pre-register the pass condition.** `hs_cosine` was named as R1's first real signal *before* R1 ran.
- **A test that supplies the input it assumes confirms the assumption, not the system.**
- **⚠️ A probe that can only report success tells you nothing when it fails.** Instrument the decision, not the happy path.
- **⚠️ stdout and stderr are two different files under SLURM.**
- **A patch that is installed is not a patch that runs.** Count the calls.
- **Two identical numbers are a finding**, stronger than a merely-large one — A7b, the vanished `forward`, A9, and the byte-identical cached/uncached GSM8K output.
- **⚠️ `min_lr` under DeepSpeed is an `AttributeError`, not a wrong number.**
- **⚠️ A dataset cache that helps at 20k documents can cost hours at 500k.**
- **`warmup_ratio` × a small `max_steps` is no warmup at all.**
- **The first step of a KDA job costs ~4 minutes of Triton autotuning.**
- **Know which log lines are per-rank and which are global.**
- **DistillKit's config has two levels**, and some features have two halves.
- **DeepSpeed's `grad_accum` wins.**
- **`re.search`, not `re.match`.** Anchor every parameter-name regex.
- **Checkpoints carry copies of the model code.**
- **A config value can become load-bearing for a reason it wasn't chosen for.** `per_device_train_batch_size: 1` was a memory decision, then a correctness precondition.
- **Library versions belong in the run log.**
- **A single-layer benchmark is not a model benchmark.** A6's 0.31× became ~0.88× at the model level.
- **Conservative estimates are better than optimistic ones.**
- **Two independent measurements agreeing is worth more than either alone.** A5b's 7.9412 and R0's 7.918.
- **Output goes where `output_path` says**, not where the last project put it.

## Sources

- `r1_stage1_hybrid31-58141472.out` / `.err` — **the stage-1 run**
- `r0_kda_smoke-58114936` (the `min_lr` crash), `-58117838` (R0), `-58131180` (R0b)
- `eval_zs_teacher_afm45b_base-58216691` — the teacher baseline
- `eval_gen_smoke_*` — the GSM8K smoke runs, cached and uncached
- `claude/test_r1_ce.py`, `claude/test_r1_ce_math.py` — held-out CE, and where the damage sits
- `main.py` / `configuration.py` / `trainer.py` / `signals.py` / `pack_logits.py` / `monkey_patch_packing.py` (DistillKit fork) — all read in full, 2026-09-18
- `test_config_4B_instruct.yaml` — the practical work's config; same packing situation, good results
- `kda_installed_0.5.2.py`, `gdn_installed_0.5.2.py` — the authoritative fla sources
- `blogpost_yaml.txt` — Arcee's stage-1 configuration, verbatim
- `RADLADS.pdf` — the three-stage pipeline, the frozen-MLP ablation, and the 0-shot / 5-shot-MMLU evaluation convention
- `claude/building_the_student.md` — Tracks A, B and C · `claude/track_b.md` — generation, closed
- `claude/startup_plan_and_pitfalls.md` — steps 5–8, the multi-node launch script, Leonardo accounting