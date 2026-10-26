# STATUS — the one page to look at first

*Updated 2026-10-18. Findings with final numbers: `claude/results.md`. Plan, chapters, schedule: `claude/thesis_plan.md` (rev 3). **Rule:** every submitted job, finished run, evaluation or decision gets one line here the same day. Detailed findings and reasoning still go into the lab notebook (`claude/first_distillkit_runs.md`), which is append-only and never pruned.*

## 1. Where things are right now

| | |
|---|---|
| Stage 1, 2×2 (R1–R4) | ✅ trained, fully evaluated (benchmarks, held-out CE/KL, arithmetic probe v2) — HumanEval ⬜ |
| R5 (1 epoch, stage 1) | ✅ done and fully evaluated (MMLU = R1, GSM8K slightly up) — ⚠️ latent damage, **not a base model** |
| Stage 2, R1–R4 | ✅ trained, **logs read: all clean**; held-out KL final R1 0.273 < R3 0.294 ≈ R2 0.297 < R4 0.330; still falling · arithmetic ✅ · `ce_outliers` ✅ clean · 0-shot + MMLU ✅ (R1-S2 MMLU 13.9 → 27.0 % rel.) · GSM8K/code 🔶 queued |
| Seed replicates R1/R2 × {43, 44} | ✅ trained, **logs read**: data-order SD 0.0006 (R1) / 0.00006 (R2) vs gap 0.0122 · evaluation ⬜ |
| R1 stage-2 LR check (cosine 5e-5 → 1e-5) | ✅ trained, **log read: worse** (0.2956 vs 0.2731; 0.528 at 100 M) · evaluation ⬜ |
| HumanEval | ⬜ script ready (`eval_lm_afm.sbatch`), smoke first |
| Math-specific run (FineMath) | ⬜ **decide ~Oct 20**; data: three splits on `$WORK/datasets` → `download_finemath.md` |
| Inference analysis (forward pass vs length) | ⬜ script after the analysis |
| Thesis text | ⬜ skeleton — 5-chapter structure in `thesis_plan.md` rev 3 |

## 2. Run registry

*Always evaluate R1 from `checkpoint-4580`: its top-level directory was overwritten by the 09-29 resume test. Stage-2 checkpoints: 1526, 3052, 4578 (≈ 300 M), 6104, 7630 (≈ 500 M), 9156, 10682, 12208, 13734, 15260 (= final).*

| run | what | job | output (under `$WORK_LUKAS/output/`) | state |
|---|---|---|---|---|
| R0 | 20-step smoke, 1 node | 58117838 | — | ✅ |
| **R1** | KDA hybrid 3:1, stage 1, 300 M | 58141472 | `r1_stage1_hybrid31/checkpoint-4580` | ✅ |
| **R2** | GDN hybrid 3:1, stage 1 | 58650590 | `r2_stage1_gdn_hybrid31/checkpoint-4580` | ✅ |
| **R3** | KDA all-linear, stage 1 | 58650592 | `r3_stage1_kda_only/checkpoint-4580` | ✅ |
| **R4** | GDN all-linear, stage 1 | 58650595 | `r4_stage1_gdn_only/checkpoint-4580` | ✅ |
| R5 | KDA hybrid, stage 1, 1 epoch (2.29 B) | 59234295 | `r5_stage1_kda_hybrid31_1ep` (top level only; ckpts 3494–31446 deleted 10-08) | ✅ ⚠️ |
| R1-s43 / R1-s44 | R1 with `training_args.seed` 43 / 44 | _fill in_ | `r1_stage1_hybrid31_s43` / `_s44` | ✅ trained |
| R2-s43 / R2-s44 | R2 with seed 43 / 44 | _fill in_ | `r2_stage1_gdn_hybrid31_s43` / `_s44` | ✅ trained |
| **R1-S2** | stage 2 on R1: KL only, nothing frozen, const 1e-5, 1 B | 59749921 | `r1_stage2_kda_hybrid31_unfreeze` | ✅ 11.83 h, clean |
| **R2-S2** | stage 2 on R2, same recipe | 59749924 | `r2_stage2_gdn_hybrid31_unfreeze` | ✅ trained |
| **R3-S2** | stage 2 on R3 | 59749930 | `r3_stage2_kda_only_unfreeze` | ✅ trained |
| **R4-S2** | stage 2 on R4 | 59749932 | `r4_stage2_gdn_only_unfreeze` | ✅ trained |
| R1-S2-cos | stage 2 on R1, cosine 5e-5 → 1e-5 | _fill in_ | `r1_stage2_kda_hybrid31_cosine5e5` | ✅ trained |
| R1-S1-math | stage 1 = R1 but FineMath (`finemath_4plus_stage1`) | 60669454 | `r1_stage1_hybrid31_finemath` | ✅ trained; 0-shot 68.4 % rel., MMLU 0.3151, GSM8K 0.152, HumanEval 0.122 · ⚠️ heavy-tailed held-out CE (6 / 14 sequences > +0.5 nat); `ce_outliers`: no R5-type activation / weight blow-up |
| R1-S2-math | stage 2 on R1 (FineWeb stage 1), FineMath (`finemath_4plus_stage2`), otherwise = R1-S2 | 60671592 | `r1_stage2_kda_hybrid31_finemath` | ✅ trained; **MMLU 0.4906 (60 % rel.)**, 0-shot 82.7 %, arithmetic 77.0, **GSM8K 0.209 (39 %)**, HumanEval 0.122, MBPP 0.146 |
| R1fm-S2-math (run B) | FineMath stage 2 on R1-S1-math, otherwise = R1-S2-math | 61289385 | `r1fm_stage2_kda_hybrid31_finemath` | ✅ trained 11.7 h, clean; eval KL 0.3056 (A 0.303); 0-shot 76.0 % rel., MMLU 0.4657 |
| R2-S2-math | FineMath stage 2 on R2 (FineWeb stage 1), = R1-S2-math otherwise; only the final checkpoint kept | 61307268 | `r2_stage2_gdn_hybrid31_finemath` | ✅ 10.7 h, clean; FineMath eval KL 0.513 → **0.3279** (R1-S2-math 0.303) |
| R3-S2-math | same on R3 | 61307269 ❌ → **61571397** ✅ | `r3_stage2_kda_only_finemath` | ✅ rerun 13.4 h, clean; FineMath eval KL 0.589 → **0.3877**; reproduces the crashed run to ≤ 0.004 |
| R4-S2-math | same on R4 | 61307271 | `r4_stage2_gdn_only_finemath` | ✅ 11.7 h, clean; FineMath eval KL 0.651 → **0.4222** |

*Stage-2 data: `datasets/fineweb_edu_3B_train_stage2_nooverlap` (split v4, 1,350,000 docs, 0 overlap with stage 1 and held-out, `provenance.json`; 49 cache files beside it).*
*⚠️ Stage-2 `eval_loss` = KL only; stage-1 `eval_loss` = composite. Same 200 docs, different quantity — never subtract one from the other.*

## 3. Evaluation matrix

✅ done · 🔶 queued · ⬜ to do · — not planned. `queue_evals.sh PLAN` names in brackets.

| model | 0-shot ×3 | MMLU | GSM8K | HumanEval | held-out CE/KL | arith v2 | ce_outliers | speed |
|---|---|---|---|---|---|---|---|---|
| teacher | ✅ | ✅ | ✅ (+100 samples) | ✅ | ✅ | ✅ | — | ✅ B1/B8/B32/B64 |
| R1 | ✅ | ✅ | ✅ (+100 samples) | ✅ | ✅ | ✅ | ✅ | ✅ B1/B8/B32/B64 |
| R2 | ✅ | ✅ | ✅ (+100 samples) | ✅ | ✅ | ✅ | — | ✅ B1/B8/B32/B64 |
| R3 | ✅ | ✅ | ✅ (+100 samples) | ✅ | ✅ | ✅ | — | ✅ B1/B8/B32/B64 |
| R4 | ✅ | ✅ | ✅ (+100 samples) | ✅ | ✅ | ✅ | — | ✅ B1/B8/B32/B64 |
| R5 | ✅ | ✅ 0.3068 | ✅ 0.0409 | — | ✅ (web/maths/Tulu CE) | ✅ (+10 ckpts) | ✅ | — |
| R1-S2 … R4-S2, R1-S2-cos (ck 15260) | ✅ | ✅ 0.3581 / 0.3268 / 0.2750 / 0.2705 / 0.3146 | ✅ (+ 2 ckpts) | ✅ (+ MBPP) | ✅ `heldout_kl` web + maths | ✅ 46.4 / 28.4 / 44.4 / 35.1 / 49.0 | ✅ clean (CE 2.4310 / 2.5021 / 2.4197 / 2.4427 / 2.4432) | (= stage 1) |
| R1-S2 ck 4578 / 7630 | ⬜ by hand (debug) | ⬜ by hand (debug) | ⬜ [s2ckpt] | — | — | ✅ all 10 ckpts (38.1 → 46.4) | — | — |
| seeds ×4 (ck 4580) | ✅ | ✅ | ✅ | — | ⬜ test_r5_kl.py | ✅ 36.8 / 38.9 · 25.2 / 27.8 | — | — |
| R1-S1-math | ✅ | ✅ 0.3151 (cloze 0.3856) | ✅ 0.152 | ✅ 0.122 (MBPP 0.114) | ✅ `heldout_kl` (web 1.009 / maths 1.437 KL) | ✅ 68.5 | ✅ tail real; not R5's mechanism | (= R1) |
| R1-S2-math | ✅ | ✅ 0.4906 (cloze 0.4099) | ✅ 0.209 | ✅ 0.122 (MBPP 0.146) | ✅ `heldout_kl` (web 0.456 / maths 0.442) — clean | ✅ 77.0 · ✅ 9 ckpts | (in `heldout_kl`) ✅ | (= R1) |
| run B (R1fm-S2-math) | ✅ 76.0 % | ✅ 0.4657 (cloze 0.3986) | ✅ 0.237 | ✅ 0.116 (MBPP 0.166) | ✅ both layouts (BOS-fragile) | ✅ 77.6 | (in `heldout_kl`) | (= R1) |
| R2-S2-math | ✅ 83.8 % | ✅ 0.4373 (cloze 0.4096) | ✅ 0.184 | ✅ 0.116 (MBPP 0.132) | ✅ both layouts (BOS-fragile) | ✅ 67.5 | (in `heldout_kl`) | (= R2) |
| R4-S2-math | ✅ 79.6 % | ✅ 0.2990 (cloze 0.4022) | ✅ 0.118 | ✅ 0.085 (MBPP 0.058) | ✅ both layouts (robust) | ✅ 66.8 | (in `heldout_kl`) | (= R4) |
| R3-S2-math | ✅ 85.8 % | ✅ 0.3079 (cloze 0.4063) | ✅ 0.174 | ✅ 0.116 (MBPP 0.074) | ✅ both layouts (robust) | ✅ 72.3 | (in `heldout_kl`) | (= R3) |

*Speed depends only on the architecture, so it is measured once on teacher + R1–R4.*
*R5's two pending jobs `eval_mml` / `eval_gsm` (59519542 / 59519848) are the MMLU/GSM8K results above — resolved 10-15.*

## 4. Open TODOs (in order) — updated 10-16

*Done since 10-15: scripts copied; logs summarised; HumanEval/MBPP smoke + 31 lprod eval jobs queued; arithmetic on all stage-2 models and seeds; `ce_outliers`; 0-shot + MMLU on stage-2 finals and seeds; FineMath data (3 splits) + stage-2 warm-up; both FineMath runs queued.*

**Must (the thesis needs these):**
1. ✅ GSM8K, HumanEval, MBPP on all queued models (results §7). ✅ HumanEval / MBPP / arithmetic / cloze on both FineMath models. 🔶 GSM8K on both.
2. ✅ FineMath runs trained and clean (logs read; `summarize_run.py` v3.2 now labels the FineMath eval set); 0-shot, MMLU letter + cloze, letter check, arithmetic done. ✅ HumanEval 0.122 / 0.122, MBPP 0.114 / 0.146 (rev 59). ✅ GSM8K 0.152 / 0.209 (rev 60). ✅ Run B trained (rev 62): 0-shot + MMLU done, below A. ✅ Run B fully evaluated (rev 66). ✅ arithmetic 77.6 (= A), ✅ `heldout_kl` both layouts. ✅ `--weighted` (rev 64): B weakest at sequence start; gap to DistillKit's 0.3056 unexplained (optional `distillkit`-style layout). ✅ `arith_stage2.py` v2: R1-S1-math 68.5, R1-S2-math 77.0. ✅ Arithmetic on R1-S2-math's 9 checkpoints (rev 60–61; exact curve from `arith_stage2.py`). ⬜ ck 1526 + 9156 rerun: their result files got terminal escape codes in the name (move `*633*` to `_misnamed/`, clean list, 1 debug job; `arith_probe.sbatch` now refuses such names).
2b. 🔶 **FineMath stage 2 on R2, R3, R4** (submitted 10-18): read the logs (`guards OK … 49 cache files`, no tokenizing, no `Froze`); then per model 0-shot, MMLU letter (+ cloze), arithmetic, `heldout_kl` on debug; GSM8K / HumanEval / MBPP on lprod.
3. ✅ **Held-out KL/CE table, web + maths** (12 models, validity OK; `results.md` §7, rev 59). ✅ R1-S1-math tail: real, model-specific, no R5-type activation / weight blow-up (rev 60). ✅ Same windows as R5? partly. ✅ `LAYOUT=docs`: tails vanish on training-format input (rev 61). ✅ **Trigger = window without BOS** (`concat_bos` removes the tails, `concat_eos` does not; rev 62). ⬜ lm-eval BOS: `add_bos_token` defaults to None → confirm it then uses the tokenizer default (adds BOS) — `sed -n 1045,1090p` of its `huggingface.py`. ✅ `LAYOUT=docs` on the default list = **table of record** (`results.md` §7; rev 63). lm-eval and the arithmetic probe prepend BOS (tokenizer default) → benchmarks are in training format. ⬜ Run B (resubmit `heldout_kl.sbatch` once trained). Optional: R1-S2 / R1-S2-math checkpoint curves via `MODELS_FILE` (web-vs-maths trade-off curve).
4. ✅ **Speed benchmark, batch 1 + 8 done** (61585206; `results.md` §8, notebook rev 74). 🔶 batch 32/64 (`DEC_BATCHES=32:64`; job 61585889 on lprod, or per-model debug jobs — resumable). ⬜ run `plot_speed.py` on the cluster (figures + `speed_table.md`). History:  scripts ready 10-20 (`speed_bench.py` 7a0cd7a1, `speed_bench.sbatch` 91a5963e, `plot_speed.py` 21f8c0bd): prefill batch 1 (forward only, 128 … 131,072 tokens) + decode batch 1 and 8 (ms/token, cache size), teacher + R1–R4 stage-1 checkpoints, exclusive node; smoke first (forward pass, batch 1, 512 … 65,536 tokens, last-token logits; teacher + R1–R4) — the efficiency claim in the title rests on it. · ✅ smoke 61585200 clean (rev 70); copy `speed_bench.py` v1.1 (64a10626) + `plot_speed.py` (391d432f) before the full job starts; optional later job `DEC_BATCHES=32:64` · `speed_bench.py` v1.2 (6ee85b73): batch > 8 uses smaller prefill chunks (32k tokens per call); the `DEC_BATCHES=32:64` job must run AFTER the main job (same result files) → `scontrol update JobId=<b32/64 job> Dependency=afterany:<main job>`
5. ⬜ Start writing: Method chapter first (pipeline, data, recipes and evaluation are frozen).

**Should (cheap):**
6. ✅ MMLU letter check (rev 51) and cloze runs R1 / R1-S2 / R3 / R3-S2 (rev 52). ✅ Cloze MMLU on teacher + all 8 students: 74–80 % retained, all students within 1.1 points (rev 53–54). ✅ Matched controls: letter MMLU 0-shot (teacher, R1, R1-S2, R3) and cloze 5-shot (teacher, R1) — letter-format loss confirmed; R1 knowledge retention 68–76 % (rev 55).
7. ⬜ Fill in seed and cosine job IDs in §2; commit scripts/YAMLs to git.
8. ⬜ Free disk (`download_finemath.md` §5) once the queued evals have read their checkpoints.

**Now (10-26):** ✅ FineMath 2 × 2 complete (R3-S2-math MBPP 0.074). ✅ Speed benchmark complete (batch 1/8/32/64, rev 77; `speed_table.md`). ✅ MBPP sample analysis (rev 78). Optional: MBPP function-name check from lm-eval samples. Then: the final summary / RQ answers, and start writing (Method chapter). *(10-20 list done: all FineMath students evaluated, `--weighted` / `--pairs`, speed batch 1 + 8.)*

**Later (remembered 10-19):** weight-swap test — R1-S2 (and R1-S2-math) with R1's original MLPs put back, then `heldout_kl` + arithmetic: how much of stage 2's gain sits in the MLP updates (CPU merge + 1 debug job).

**Only if time remains:** FineMath stage 1 → FineMath stage 2 (run B) if R1-S1-math beats R1 on arithmetic; retrieval probe.

## 5. Decisions (dated)

| date | decision | why / where |
|---|---|---|
| 09-09 | Corpus: FineWeb-Edu (public), not Arcee's internal mix | `startup_plan_and_pitfalls.md` step 3 |
| 09-18 | Stage 1 = Arcee's merged RADLADS 1+2, MLPs frozen, 300 M tokens | notebook rev 13 |
| 09-21 | Document isolation closed: train with packing, no isolation, like the reference | notebook Part 4 |
| 09-25 | Mechanism × layout 2×2 (KDA/GDN × hybrid/all-linear) | notebook rev 21 |
| 10-07 | R1–R4 stay the stage-1 models; R5 is not a base (latent damage) | notebook rev 36–37 |
| 10-08 | Stage 2 = RADLADS step 2 (KL only, nothing frozen, const 1e-5, 1 B) on all four | notebook rev 39–42 |
| 10-08 | Zero-overlap stage-2 data, duplicates skipped (split v4) | notebook rev 42 |
| 10-08 | Seed replicates: R1, R2 × seeds 43, 44 (data order only) | notebook rev 42 |
| 10-09 | R1 stage-2 LR check (cosine 5e-5 → 1e-5, after HALO) | notebook rev 42 |
| 10-09 | Cut: FLP, de-novo, corpus mix, IFEval (converted students), long-context fine-tune, RoPE in linear layers, layer selection | notebook rev 42 |
| 10-12 | Lab notebook archived; this STATUS page introduced | — |
| **10-15** | **Meeting:** one unified thesis (no Part I/II split; practical work contributes only chosen experiments); 5 chapters: Intro → Background/Related → Method → Experiments/Results → Conclusion/Future work | `thesis_plan.md` rev 3 |
| **10-15** | RQs fixed only after the experiments (~Oct 22); brainstorm in `thesis_plan.md` | `thesis_plan.md` rev 3 |
| **10-15** | Inference analysis = forward-pass runtime vs sequence length (KLA paper App. E.3), batch 1; lengths up to 64k, not ≤ 2048 | `thesis_plan.md` rev 3 |
| **10-15** | HumanEval (base `humaneval`, 0-shot, pass@1) on teacher, R1–R4 stage 1 and the stage-2 finals; MBPP optional | notebook rev 43 |
| **10-15** | Math-specific distillation (FineMath, same pipeline) — decision after the stage-2 results, ~Oct 20; data prepared now | `thesis_plan.md` rev 3 |
| **10-15** | No fine-tuning of any kind (out of scope, no compute) | meeting |
| **10-15** | Queue the whole evaluation battery at once; eval jobs at 8 CPUs (not 32: billed as a full node) | notebook rev 43 |
| 10-09 → 10-15 | ~~Retrieval probe~~ → only if time remains after the inference analysis | `thesis_plan.md` rev 3 |
| **10-16** | Stage-2 recipe of record = **constant 1e-5** (cosine 5e-5 → 1e-5 lost by 8 % held-out KL at 1 B) | notebook rev 44 |
| **10-16** | 0-shot trio run by hand on the debug queue; MMLU on the debug queue (< 30 min); HumanEval/MBPP queue decided after a timed smoke run | notebook rev 44 |
| **10-16** | MBPP uses lm-eval's task default (3 fixed prompt examples, Austin et al. 2021) — no `--num_fewshot` passed; confirmed from the installed yaml | notebook rev 44–45 |
| **10-16** | Nothing is queued on debug (jobs beyond 2 are cancelled): 0-shot + MMLU by hand; everything else via `queue_evals.sh` on lprod | notebook rev 45 |
| **10-16** | FineMath: 3.2 M docs, all on `$FAST`; one pool unless two runs are stacked | notebook rev 45 |
| **10-16** | `queue_evals.sh all` → 31 lprod jobs (GSM8K, HumanEval, MBPP); arithmetic + ce_outliers by hand on debug (`queue_evals.sh debug` prints the lines) | notebook rev 46 |
| **10-16** | Arithmetic probe on stage-2 finals, R1-S2 curve, seeds ✅ — R1-S2 46.4 % (pre-reg ≥ 43.3 met), gap closure only 0.15 → **math (FineMath) stage 2 indicated** | notebook rev 47 |
| **10-16** | **Queued:** 31 eval jobs, IDs 60658287–60658339 (GSM8K 5 finals + 2 ckpts + 4 seeds; HumanEval + MBPP on 5 finals and teacher + R1–R4) | — |
| **10-16** | `ce_outliers` on the 5 stage-2 finals ✅: no catastrophic sequences, kernel-independent, small weight drift; web CE gap closed only 3–16 % | notebook rev 48; `results.md` §7 |
| **10-16** | 0-shot + MMLU on the 5 stage-2 finals and 4 seeds ✅: constant-LR stage 2 lifts MMLU in the hybrids (R1 13.9 → 27.0 % rel., R2 11.5 → 19.2 %), not all-linear; cosine only 16.1 % → **constant 1e-5 settled**; which stage-2 component causes the lift is confounded; still far from Arcee (79–81 % rel.); 0-shot mechanism gap robust across seeds (+6.0 ± 1.1) | notebook rev 49–50; `results.md` §4, §6, §7 |
| **10-16** | FineMath runs submitted: stage 1 (60669454), stage 2 on R1 (60671592); eval set = `finemath_4plus_heldout`; stage-2 warm-up 60669372 ✅ (1.168 B tokens) | `download_finemath.md` |
| **10-16** | FineMath data **supersedes the `$FAST` / one-pool line above**: three splits (held-out 2,000 / stage 1 ≥ 300 M / stage 2 ≥ 1 B effective tokens), disjoint by text, in `$WORK_LUKAS/datasets` like FineWeb-Edu | `download_finemath.md` |
| **10-18** | `heldout_kl` (web + maths, 12 models) ✅: FineWeb students 2.8–3.5× further from the teacher on maths; FineMath stage 2 trades web KL +56 % for maths KL −45 %; **R1-S1-math heavy-tailed** (R5-like) → per-sequence check + `ce_outliers` before run B is interpreted. Code: FineMath HumanEval 0.122 (49 % of teacher) | notebook rev 59; `results.md` §7 |
| **10-18** | **FineMath stage 2 on R2, R3, R4, each from its FineWeb stage 1** (= R1-S2-math's design; completes the 2×2 under FineMath); only the final checkpoint kept; mixed-corpus run not decided | notebook rev 60 |
| **10-18** | R2/R3/R4-S2-math submitted. **Mixed-corpus stage 2 (50/50 FineWeb-Edu + FineMath on R1): raise at the next meeting**, not run now. Speed benchmark after the FineMath runs | notebook rev 61 |
| **10-19** | Held-out table of record = training format (`LAYOUT=docs`); the concatenated table is kept as a robustness test. Run B ≈ A on arithmetic, worse elsewhere → recipe of record stays FineWeb stage 1 + FineMath stage 2 | notebook rev 63 |
| **10-19** | Wait for R4-S2-math, then evaluate R2/R3/R4-S2-math together; then the speed benchmark | — |
| **10-26** | **Experiments closed** (stage 2 web + FineMath ×4, run B, speed B1–B64, MBPP samples). Thesis plan rev 4: RQs proposed, 4-week writing schedule. Optional ≤ 1-day weight-swap test and the mixed-corpus run → decide at the next meeting (default: weight-swap optional, mixed corpus = future work) | `thesis_plan.md` rev 4 |

## 6. Where to find what

| topic | document → section |
|---|---|
| **All results, current truth, one entry each** | **`results.md`** |
| **Plan, RQ brainstorm, chapters, schedule, reading list** | **`thesis_plan.md`** |
| Leonardo, DeepSpeed, multi-node, budget, file systems | `startup_plan_and_pitfalls.md` |
| FineMath download, splits, disk, cleanup | `download_finemath.md` |
| KDA student: construction, weight transfer, initialisation, A-tests, memory (B9a) | `building_the_student.md` |
| Generation / cache bridge (Track B), memory curve | `track_b.md` |
| GDN student (Track C), parameter decomposition, KDA vs GDN gates | `building_GDN_student_track_c.md` |
| R1 results, Arcee comparison, KL 0.314 vs 0.2 | notebook Part 1: "R1 worked", "Can we compare to the blog post?" |
| GSM8K: arithmetic circuit, loop/termination table | notebook: "RESOLVED — the GSM8K collapse is the ARITHMETIC CIRCUIT" |
| Off-distribution CE gaps (web / maths / Tulu 3) | notebook: "What stage 1 did NOT recover" |
| 2×2: benchmarks, held-out CE/KL, layout vs context length | notebook Part 2: "The complete 2×2", "Held-out CE, KL and hs_cosine" |
| Arithmetic probe v2, tokenizer, problem-space pattern | notebook: "R5 diverged … what the probes found" |
| R5: barrier crash, divergence, latent damage, MMLU/GSM8K | notebook Part 2, R5 sections; rev 33–37, 43 |
| Stage-2 design, split, warm-up, R1-S2 log | notebook rev 38–43 |
| Config settings and why; token accounting | notebook Part 3 |
| Traps | notebook Part 5 (+ trap lists in each companion doc) |
| Scripts: eval (`eval_lm_afm.sbatch`), queueing (`queue_evals.sh`), log summary (`summarize_run.py` v3) | project `claude/` |