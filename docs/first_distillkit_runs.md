# First DistillKit Runs — Stage 1 with the KDA Student

*⚠️ **From revision 42 this document is the LAB NOTEBOOK (archive).** It keeps the complete reasoning history, including every retracted explanation, and nothing is deleted from it. **Current state lives elsewhere:** `claude/STATUS.md` = run registry, open TODOs, decisions, where to find what · `claude/thesis_plan.md` = research questions, chapters, schedule. Items that were open here and are now settled are marked **✅ RESOLVED** in place and listed in the index under "How to read this document".*

*Companions: `claude/startup_plan_and_pitfalls.md` = infrastructure (closed). `claude/building_the_student.md` = the model, Tracks A/B/C. `claude/track_b.md` = generation, closed. `claude/building_GDN_student_track_c.md` = the GDN student, closed.*

*Revision 78 (2026-10-26) — **MBPP samples over all 500 problems (`mbpp_samples.py`, FineMath stage-2 students + teacher).** Rev 75's hypothesis (all-linear students fail to copy the function name) is **falsified**: names are right in 92–93 % (hybrids 99 %, teacher 100 %). The all-linear failure is termination and degeneration: `[BEGIN]` instead of `[DONE]` 17 / 26 % (R3 / R4; others ≤ 0.2 %), cut by the token limit 24 / 50 % (teacher 2 %), repetition loops 25 / 54 % (teacher 1.8 %), valid Python 73 / 46 % (teacher 98 %). Cutting at the first `[BEGIN]` recovers only part of the gap (7.4 → 9.6, 5.8 → 9.2; hybrids 14.6 / 13.2). ⭐ The examples' `\r\n` line endings are reproduced by the teacher in 89 %, R2 68 %, R1 33 %, all-linear **0 / 500** — a clean marker of verbatim in-context copying, which only the models with softmax attention do. Joins lettered MC: the skills lost without attention are the ones read off the prompt (format, delimiters, exact tokens), plus a higher tendency to loop in long greedy generation. Hybrid vs teacher (14.6 vs 33.2): same format behaviour, weaker code.*

*Revision 77 (2026-10-26) — **Speed benchmark complete; the batch-32/64 OOMs re-measured with v1.2.** The cluster copy had been v1.1 (`grep -c 'CHUNK \* 8'` = 0); with v1.2 the false OOMs become measurements: teacher B32 × 16k 98.2 ms (36.9 GB), B64 × 4k 63.1, × 8k 110.7; hybrids B32 × 64k 98.9 / 98.6 (38.0 GB), B64 × 16k 66.2 / 65.9, × 32k 113.7 / 113.5 (39.2 GB). All remaining OOMs are real (cache > 64 GB) and now carry the `[during cache build, chunk …]` tag. ⭐ **Decode time is a function of the cache bytes read per step, the same for teacher and hybrids** (~37–39 GB → 98–114 ms in both) — the hybrid simply needs 4× more tokens to reach the same bytes; capacity on one A100: teacher ≈ 0.5M cached tokens, hybrid ≈ 2.1M, all-linear unlimited by the cache. The 'R1 got slower' impression: new points are much longer batch × context than the earlier ~47 ms ones; at matched points debug and lprod agree. Debug jobs were exclusive (sbatch `--exclusive`; empty GPU process list). Single slow steps at B64 × 32k (848 / 884 ms max, near-full memory) leave the medians unaffected. MBPP samples (first 10 per model, two of four files identifiable only by pattern): every model copies the function name correctly → the 'wrong name' hypothesis (rev 75) is not supported on these; the difference is termination and verbatim copying — two files reproduce the examples' `\r\n` line endings in 100 % of answers and never write `[BEGIN]`, two never copy `\r\n` and start a new `[BEGIN]` example instead of `[DONE]` in 20–30 %, with repetition loops in 30–40 %. `mbpp_samples.py` counts this over all 500 problems.*

*Revision 76 (2026-10-26) — **Speed, batch 32 / 64 (debug).** All-linear students flat at 53–54 (R3) / 51–52 ms (R4) up to 64 × 128k = 8.4 M cached tokens (state 1.5 / 3.1 GB) → ~1,200 tok/s at any length; hybrids flat to B32 × 16k, 57 ms at B32 × 32k; teacher 30 ms until ~128k cached tokens, then rising (+≈ 0.16 ms per 1k): the teacher's decode time is a function of batch × context, so the crossover moves from > 128k (B1) to ~16–32k (B8) to ~4–8k (B32). GDN 3–5 % faster than KDA at every batch size. ⚠️ **Several OOMs are artefacts:** the failed allocations (4.5 GiB at B32, 9 GiB at B64, independent of T) equal B × 4,096 × 18,432 × 2 bytes — the MLP activation of a 4,096-token-per-row chunk, which v1.2 had replaced by 1,024 / 512 tokens per row; the untagged error text (v1.2 tags cache-build failures) confirms the cluster ran the v1.1 chunking. Affected: teacher B32 ≥ 16k and B64 ≥ 4k, hybrids B32 64k and B64 ≥ 16k; true OOMs (cache alone > 64 GB): teacher B32 ≥ 32k, B64 ≥ 16k, hybrids B32 128k, B64 ≥ 64k. Measured times unaffected. Re-measure the affected points with v1.2.*

*Revision 75 (2026-10-26) — **R3-S2-math MBPP 0.074 (job 61668012) — the FineMath 2 × 2 is complete.** MBPP splits by layout, not mechanism: hybrids 0.146 / 0.132, all-linear 0.074 / 0.058 (hybrid − all-linear z ≈ 3.7 KDA, 4.0 GDN; KDA − GDN z ≤ 1). HumanEval shows no layout effect (R3 0.116 = R2). Hypothesis: MBPP needs the function name copied from the test asserts plus a 3-shot `[BEGIN]…[DONE]` format, HumanEval only continues a given signature — in-prompt copying is the attention-dependent skill, as with lettered MC. Untested; a sample check (function name ≠ the name in the asserts) would decide it.*

*Revision 74 (2026-10-26) — **Speed benchmark (61585206, batch 1 + 8) and GSM8K / code for R2/R3/R4-S2-math.** (1) **Prefill:** students overtake the teacher between 8k and 16k (crossover ≈ 10–12k); at 128k hybrids 1.96 / 1.98×, all-linear 2.90 / 2.95× faster; log-log slope 64k → 128k teacher 1.67, hybrids 1.29, all-linear 1.00. Below 8k the students are slower (fixed per-call cost of the chunked kernels; 0.33–0.49× at 128 tokens). Prefill memory identical for all five (MLP activations). (2) **Decode:** batch 1 — the teacher stays faster up to 128k (31 → 41.5 ms vs flat 44–52 ms; launch-bound HF eager, as rev 70 said). Batch 8 — the teacher rises from 16k (35.6 → 56.3 → 97.6 ms) and OOMs at 128k (~74 GB cache); students flat 46–54 ms → 1.8–2.1× at 64k, throughput 82 vs 148–175 tok/s. Cache at 128k: 9.2 GB vs 2.3 GB (hybrids, 3.9×) vs 57 MiB constant (all-linear, 163×); break-even ≈ 800 tokens (B9a 670). GDN 4–5 % faster in decode, ≤ 2 % in prefill. All checks passed (chunked = one-shot, argmax same; drift ≤ 0.1 %). (3) **GSM8K under FineMath:** R2 0.184, R3 0.174, R4 0.118 (R1 0.209): 22–39 % of the teacher vs ≤ 7 % after FineWeb stage 2. **The all-linear students recover the answer format** (flex/strict 1.03 / 1.10, was 2.37 / 28.4) → the FineWeb-era "hybrid keeps the format" was a corpus effect; lettered MC stays the one attention-dependent skill. Mechanism on GSM8K: hybrid +0.025 (z ≈ 1.6), all-linear +0.056 (z ≈ 4); layout: KDA +0.035 (z ≈ 2.3), GDN +0.066 (z ≈ 4.7) — R4 ≈ R2 on arithmetic but well below on GSM8K. HumanEval 0.085–0.122 (no ordering), MBPP R2 0.132, R4 0.058. ⚠️ The "R3 MBPP" output sent with these was `r3_stage1_kda_only/checkpoint-4580` (0, known); R3-S2-math MBPP (61668012) still owed.*

*Revision 73 (2026-10-21) — **Weighted and paired checks over all FineMath models.** (1) `--weighted`: only the two BOS-fragile models (R2-S2-math, run B) have a start-of-sequence weakness (web KL on short documents 0.50 / 0.95 vs long 0.41 / 0.71; all others within ±0.03), so per-document means inflate exactly those two. Token-weighted maths ratios — layout 1.32 (KDA) / 1.23 (GDN), mechanism 1.17 / 1.09 — sit closer to DistillKit's own eval than the per-document ones. **Decision: the thesis reports token-weighted training-format KL**; per-document is the robustness view. (2) `--pairs`: mechanism GDN − KDA under FineMath z +11 to +15, KDA closer on 63–64 of 64 documents in both corpora and layouts; layout clear for KDA (z +7 web / +10 maths) and for GDN on maths (z +4.1); GDN's web sign (−0.018, z −2.2) is R2-S2-math's short-document weakness. Corpus: FineMath stage 2 is closer to the teacher on **all 64** maths documents for all four students (−0.53 to −0.67, z ≈ −11) and further on 62–64 web documents (+0.11 to +0.18).*

*Revision 72 (2026-10-21) — **R3-S2-math: arithmetic and held-out KL; the FineMath 2 × 2 is complete except GSM8K / code.** Arithmetic **72.3** (+27.9 vs R3-S2) — the corpus effect holds in all four students (+30.6 / +39.1 / +27.9 / +31.7). Mechanism under FineMath: KDA − GDN +9.5 (hybrid), **+5.5 [+3.7, +7.3] (all-linear)** — about half the FineWeb gaps, still above the noise floor. Layout under FineMath: all-linear − hybrid **−4.7 [−6.5, −2.9] (KDA)**, −0.7 (GDN): arithmetic comes back largely without attention (R3-S2-math = 94 % of R1-S2-math), lettered MC does not (rev 71). Held-out, training format: web 0.379 / maths 0.388 (KDA layout ratio on maths 1.27, = DistillKit's 1.28); concatenated: 0 / 2 bad windows → **neither all-linear student is BOS-fragile**, two of three FineMath hybrids are — the attention-sink reading gains support. FineMath stage 2 costs web KL +47 to +68 % and gains maths KL −59 to −63 % in every student.*

*Revision 71 (2026-10-21) — **R3-S2-math (rerun 61571397) trained and partly evaluated.** Clean, 13.4 h, cache hit, no spikes; FineMath eval KL 0.589 → **0.3877**. Its first 500 M reproduce the crashed attempt to ≤ 0.004 KL (same YAML and seed) — a free measurement of run-to-run nondeterminism, far below every reported effect. Layout cost on the FineMath eval is the same for both mechanisms (all-linear/hybrid 1.28 KDA, 1.29 GDN); mechanism GDN/KDA 1.09 all-linear, 1.08 hybrid. Benchmarks: 0-shot 0.5256 / 0.7239 / 0.6796 = **85.8 % rel.** (best of all students; R3-S2 84.8 %); **letter MMLU 0.3079** (14.5 % rel.; R3-S2 0.2750, +3.3); cloze 0.4063 (R3-S2 0.4034). ⭐⭐ **The letter-format result now holds in both mechanisms:** FineMath restores lettered MC in the two hybrids (+13.3, +11.1) and not in the two all-linear students (+3.3, +2.9), with knowledge (cloze 0.40–0.41) and general ability unchanged — the cleanest layout effect in the project. Arithmetic, GSM8K / code and held-out KL for R3-S2-math queued. Speed: `speed_bench.py` v1.2 shrinks the prefill chunk above batch 8 (an OOM must come from the cache, not from MLP activations); the batch 32/64 job writes into the same result files as the main job and must run after it.*

*Revision 70 (2026-10-21) — **Speed smoke test (61585200, debug, 3 lengths): pipeline works; two findings.** (1) **Prefill crossover already between 4k and 16k:** at T = 16,384 the teacher needs 857 ms, the students 792–817 ms (1.05–1.08×; R3 KDA all-linear fastest); at 4,096 the students are 0.90–0.93×, at 512 much slower (67–97 vs 37 ms: fla's chunk kernels carry a fixed per-call cost). Teacher throughput already falls between 4k and 16k (21.4k → 19.1k tok/s), the students' still rises. Prefill peak memory is the same for all five (MLP activations dominate) — the conversion buys nothing in prefill memory, as B9a said. (2) ⚠️ **Batch-1 and batch-8 decode in HF eager mode is launch-bound, not memory-bound:** teacher 31 ms/token, students 47–55 ms/token, all flat up to 16k (only the teacher at batch 8 starts rising: 31 → 35.5 ms at 16k). Reading the ~9–10 GB of weights takes ~6 ms on an A100, so ~80 % of a step is Python/kernel-launch overhead — and a KDA/GDN layer launches more, smaller kernels than a FlashAttention layer (projections, three short convolutions, gate bottlenecks, the recurrent kernel, gated norm, cache bookkeeping). So at these lengths decode speed measures framework overhead; the teacher's growing KV cost only shows at large batch × context. Cache sizes as predicted: teacher 72 KiB/token (36.5 / 288.5 / 1,152.5 MiB), hybrids 51.6 / 114.6 / 330.6 MiB, all-linear constant 56.6 MiB (B9a's exact tensor sum 47.1 MiB; the rest is allocator granularity). (3) **The chunked-prefill check was mis-designed, not failed:** R1's step diff 17.7 came from a near-tie at the last prompt position (prefill diff 0.17 = 2–3 ulp, like every model; random-token prompts give flat distributions, logit scale ~9), after which v1 fed each path a *different* token. v1.1 feeds both paths the same token and records the top-2 gap. R1's large `prefill_with_cache_s` (12–24 s) is first-use Triton compilation of the KDA kernels in a fresh `TRITON_CACHE_DIR` (R3, run after R1, shows none); the timed medians are after warm-up and unaffected. `DEC_BATCHES` (e.g. `1:8:32:64`) can add larger decode batches in a later job; existing points are skipped.*

*Revision 69 (2026-10-20) — **Speed benchmark designed and written (`speed_bench.py`, `plot_speed.py`).** Protocol after KLA App. E.3, extended: (1) **prefill** = one forward pass, batch 1, no cache, T = 128 … 131,072 (2^7–2^17), median of 10 (5 at 128k) after 3 warm-ups, CUDA events, peak memory above weights; (2) **decode** = chunked prefill (4,096 tokens/row/call) with the cache, then 32 greedy steps after 4 warm-ups, ms/token (CUDA events per step + host wall-clock), cache size as allocated memory, batch 1 and 8; OOM recorded as a result. Identical for every model: bf16, `flash_attention_2` for all softmax layers, fla 0.5.2 kernels (chunk; fused_recurrent for ≤ 64 new tokens), `use_cache` explicit on every call (checkpoints store `false`), `logits_to_keep=1`, no `attention_mask` (no padding → avoids fla's unpad copy and `is_segment_mask`'s GPU sync), random ids with BOS (runtime is content-independent), HF DynamicCache for all attention layers, no compile/CUDA graphs. Built-in checks: chunked = one-shot prefill (T 4096, teacher as bf16 control), drift re-measurement at T 8192. **Student construction reviewed for speed:** correct (A3/B6/B7/C1), no artificial slow-down; the costs are architectural and deliberate — full-width k/v in every linear layer (fla KDA has no grouped keys; Arcee did the same: +0.38–0.62 B parameters, 5.00–5.24 B vs 4.62 B), the 9 attention layers' sigmoid gate, GDN's full-rank output gate. Not used: KDA's `safe_gate` fast path (would change the trained function) — a possible underestimate of KDA speed. Students' attention layers are NoPE. All speed numbers are cost only; long-context quality is untested (trained at 2,048). Stage-1 checkpoints stand for all runs (speed depends only on architecture).*

*Revision 68 (2026-10-20) — **FineMath stage 2 on R2 and R4 evaluated (R3 rerun 61571397 pending; first attempt: NCCL "remote process exited or network error" at step 8,550 = infrastructure).** (1) Corpus effect in every student: arithmetic R2 28.4 → **67.5** (+39.1), R4 35.1 → **66.8** (+31.7) — pre-registered ≥ +15 ✅. (2) **Mechanism under FineMath: KDA − GDN (hybrid) +9.5 [+7.6, +11.4]** — half of FineWeb's +17.9, still above the noise floor (pre-registered ✅); DistillKit eval KL GDN/KDA 1.08 = FineWeb's 1.086. (3) **Layout under FineMath (GDN): all-linear = hybrid on arithmetic** (−0.7 [−2.6, +1.3]; FineWeb +6.7). (4) ⭐⭐ **Lettered-MC MMLU needs attention layers:** FineMath lifts it in R2 (0.327 → **0.437**, as R1 +13.3) but hardly in R4 (0.271 → **0.299**, 12 % rel.), although R4-S2-math equals R2-S2-math in arithmetic (66.8 vs 67.5) and in cloze knowledge (0.402 vs 0.410). Same knowledge, same procedure — only the in-context mapping of options to letters is missing; the cleanest layout dissociation in the project. (5) 0-shot: R2-S2-math 83.8 % (R2-S2 81.0 %), R4-S2-math 79.6 % (79.3 %) — FineMath stage 2 costs R1 −3.7 but not the GDN students; HellaSwag −1 to −1.6 raw in all three. (6) Robustness: R2-S2-math BOS-fragile (14 / 24), R4-S2-math not (0 / 2) — the fragility shows up only in hybrids, consistent with a softmax-attention first-token sink. Per-document training-format ratios for fragile models are inflated (R4/R2 maths 1.13 there vs DistillKit 1.29); `--weighted` over all FineMath models before quoting. GSM8K / code for R2, R4 queued.*

*Revision 67 (2026-10-20) — **R2/R4-S2-math trained clean; R3-S2-math FAILED.** R2-S2-math: 10.7 h, cache hit, no spikes, FineMath eval KL 0.513 → 0.3279 (R1-S2-math 0.303 → GDN/KDA 1.08, the same ratio as after FineWeb stage 2, 1.086). R4-S2-math: 11.7 h, clean, 0.651 → 0.4222 (all-linear/hybrid for GDN on maths 1.29; FineWeb stage 2 on web 1.11 — the layout cost is larger on maths, as in the training-format table). ⚠️ The 1.23 GDN/KDA from the per-document training-format table (rev 63) disagrees with DistillKit's 1.08: as for run B, per-document means can inflate a start-of-sequence weakness; `--weighted` owed before quoting it. **R3-S2-math died at ~step 8,550 (≈ 560 M tokens) with an NCCL watchdog timeout** (6 timeout lines, 3 tracebacks, srun exit 1, not stamped; eval 0.589 → 0.444 by 500 M, healthy until then) — most likely a node or network fault, not the run. Weights-only checkpoints cannot be resumed with optimizer state, and restarting from `checkpoint-7630` would replay the same data from the start, so **rerun from scratch** with the identical YAML (same seed → same data order); old directory kept as `_failed61307269` until the rerun is done. (Correction: the 10-19 note that R3 "finished successfully" was wrong — it had left the queue because it crashed.)*

*Revision 66 (2026-10-20) — **Run B fully evaluated:** GSM8K strict 0.2373 (= flexible; 44 % of the teacher), HumanEval 0.116, MBPP 0.166 — no established difference from A (0.209 / 0.122 / 0.146; z ≤ 1.7). So the stage-1 corpus does not matter for the procedural skills once stage 2 is FineMath; it only costs general ability (0-shot 76 vs 83 %, letter-MMLU 0.466 vs 0.491, cloze 0.399 vs 0.410) and adds start-of-sequence fragility. Recipe of record unchanged. Flex = strict means the `#### N` answer format is produced every time; remaining errors are wrong numbers. Weight-swap test (R1-S2 with R1's MLPs) noted for later.*

*Revision 65 (2026-10-19) — **Run B cloze MMLU 0.3986** (A 0.4099, R1 0.3976): the stage-1 corpus does not change stored knowledge either. **Paired significance in training format** (`LAYOUT=docs heldout_kl_seqs.py --pairs`): layout (all-linear − hybrid) z +4.1 to +9.0 in all 8 comparisons, mechanism (GDN − KDA) z +8.1 to +13.7 in all 8 — both effects are real on these 64 documents; the concatenated-layout layout z-values of rev 59 (web −2.4 / −3.2) are withdrawn.*

*Revision 64 (2026-10-19) — **Token-weighted check on run B; summaries written.** `heldout_kl_seqs.py --weighted` (training format): B maths KL per-document 0.503 → token-weighted 0.441 (A 0.321), long documents 0.434 (A 0.311), **short documents (< 512 tokens) 0.709 (A 0.285)**; web short 0.949 vs long 0.714 (A 0.370 / 0.346). Weighting explains only part of the gap to DistillKit's own 0.3056: B is weakest at the start of a sequence even with BOS — the milder form of its no-BOS failure — and DistillKit's eval (different 200 documents, ~2 documents per packed row, so most tokens have long context) dilutes that. A, R1-S2 and R1 show no short/long difference worth noting. B's maths standing is therefore "= A on arithmetic and DistillKit's eval, worse in our per-document check"; not pursued further. R3-S2-math finished; R4 pending; all three evaluated together, then the speed benchmark. Summary table of stage 2 and the FineMath runs added to `results.md` §7.*

*Revision 63 (2026-10-19) — **Training-format held-out table on every model; run B arithmetic; complete FineMath curve.** (1) `LAYOUT=docs` on the default list. It reproduces DistillKit's own eval where comparable (GDN/KDA 1.098 / 1.120, all-linear/hybrid 1.057 / 1.078 vs DistillKit 1.086 / 1.123 and 1.075 / 1.112; R1-S2-math maths 0.3045 vs 0.303), so it becomes the **table of record**; the concatenated table stays as a robustness test. In training format: stage 2 is uniform (web −20/−21 %, maths −16/−17 % for all four); **KDA < GDN in 8 / 8** (+9–14 %); **hybrid < all-linear in 8 / 8** (+4–8 % web, +18–20 % maths) — this **replaces** rev 59/60's "layout is corpus-dependent / all-linear ahead on web": that sign came from windows without BOS, which hurt the hybrids (softmax attention, first-token sink) and not the all-linear students. Rev 59's "R2 gains least in stage 2" is the same artefact. R5 = R1 (0.307 vs 0.304). FineMath stage 2 trades web for maths in GDN too (R2-S2-math web +68 %, maths −59 % vs R2-S2); within FineMath stage 2 GDN trails KDA by ~24 % (FineWeb: ~10 %). (2) **Run B is the most BOS-fragile model** (concatenated: KL 2.7 / 2.8, 33 / 40 bad windows) — a FineMath stage 2 did not repair R1-S1-math's fragility, it amplified it; R2-S2-math has a milder form (14 / 24), R1-S2-math almost none. In training format B is also worse than A (web 0.80 vs 0.35; maths 0.50 vs 0.30) — but DistillKit's own eval says 0.3056 vs 0.303 on maths. Suspect: per-document means weight short documents / early positions, where B may be bad; `heldout_kl_seqs.py --weighted` (token-weighted, short vs long documents) owed. (3) **Arithmetic: B 77.6 = A 77.0** (+0.6 [−1.1, +2.2]); the stage-1 corpus does not matter for arithmetic once stage 2 is FineMath. (4) FineMath curve complete: 52.9 at 100 M (+14.9 over FineWeb at equal tokens), 71.5 at 600 M; slope +2.22 vs +0.76 per 100 M after 300 M. (5) lm-eval (`add_bos_token` None → tokenizer default) and `arith_probe.py` (`tok(prompt)`) both prepend BOS: **all benchmark numbers are in training format** and unaffected by the fragility. (6) ⚠️ R3-S2-math is not in the queue and had no `checkpoint-15260` when `heldout_kl` ran — check whether it failed.*

*Revision 62 (2026-10-19) — **The trigger is a missing BOS; run B trained; scripts checked against DistillKit's loss code.** (1) `heldout_kl.py` v3, one change at a time: with every window starting with BOS (`concat_bos`) the tails of R1-S1-math and R5 disappear (R1-S1-math worst +0.63 / +0.46, R5 ≈ R1); with EOS added at every junction but windows still starting mid-text (`concat_eos`) they stay (R1-S1-math 11 / 9 bad, worst +3.5; R5 worst +6.2 / +7.3). Teacher, R1, R1-S2-math are unaffected in every layout. So rev 37's "latent damage" of R5 = **breaking on some sequences that start without BOS**; plausibly the first-token / attention-sink role (R1 carries the usual large early activation on such a window, R1-S1-math none, R5 an excessive one) — not tested. ⚠️ **Correction to rev 61:** R1-S1-math does not *forget* web text — it was never aligned on it (its stage 1 starts from the converted init, like R1, but trains only on FineMath); its larger web gap in training format (+0.30 vs R1 +0.12) is weaker transfer from maths-only alignment. Forgetting applies only to R1-S2-math (web KL 0.304 → 0.355 in training format). (2) **Run B** (FineMath stage 1 → FineMath stage 2, 61289385): clean, 11.7 h, cache hit; FineMath eval KL 0.402 → 0.3056 vs run A 0.468 → 0.303 (B starts ahead, same end point); 0-shot 0.4991 / 0.6991 / 0.6448 = 76.0 % rel. (A 82.7 %, R1-S1-math 68.4 %); MMLU 0.4657 (A 0.4906). A FineMath stage 1 buys nothing after a FineMath stage 2 and costs general ability → **FineWeb stage 1 + FineMath stage 2 is the recipe of record**; B's arithmetic / GSM8K / code / held-out KL owed. (3) **Formula check** against `kl.py`, `cross_entropy.py`, `hidden_state.py`, `trainer.py`, `main.py`: our KL = DistillKit's dense KL (fp32 log-softmax at T, KL(teacher ‖ student) summed over vocab, × T², token mean over all valid tokens); CE = HF causal-LM loss as `cross_entropy.py` returns it; stage-1 total = Σ wᵢ·lossᵢ / Σ wᵢ (weights sum to 1). Negligible differences: we drop the last position, average per-sequence means, and do not cut logits to the tokenizer vocabulary. Cross-check: R1-S2-math training-format maths KL 0.3045 vs DistillKit's final eval 0.303. Detail worth knowing: `layer_mapping: all` maps hidden_states[0..35], i.e. the embedding output and layers 0–34 — the last layer's output is not in the hs_cosine term.*

*Revision 61 (2026-10-18) — ⭐⭐ **The catastrophic sequences of R1-S1-math AND R5 are a format-shift fragility, not damage on ordinary input.** `heldout_kl.py` v2 `LAYOUT=docs` (each sequence one document from its start, `[BOS] text [EOS]`, as DistillKit trains and evaluates): R1-S1-math maths KL 1.437 → **0.430**, CE gap **+0.027** (closest to the teacher of any model), 0 bad sequences (was 14); web worst +1.18 (was +5.39). **R5: web KL 0.307 = R1's 0.304**, no bad sequence (was 4); maths ≈ R1. The concatenated layout differs from training in two ways — windows start mid-document without BOS, and junctions are `text<BOS>` without EOS — and only these two longer/hotter stage-1 models break on it; R1, all stage-2 models and the teacher do not. This **qualifies rev 37**: R5's damage is real but input-format-triggered (the "mean hides it" lesson stands; "unusable" becomes "fragile under format shift"). Owed: `LAYOUT=concat_bos` / `concat_eos` (v3) to name the trigger; whether lm-eval prompts (no BOS by default) hit it. Plain forgetting separates from the tail: in training format R1-S1-math's web gap is +0.30 (R1 +0.12, R1-S2-math +0.16) — FineMath stage 1 forgets web text far more than FineMath stage 2. R5 shares 2 / 6 web and 6 / 14 maths bad windows with R1-S1-math. **Exact arithmetic curve** (`arith_stage2.py` v3): FineMath − FineWeb at equal tokens +19.6 (300 M) → +31 (800 M–1 B), slope +2.25 vs +0.76 per 100 M after 300 M; ck 1526 / 9156 missing from the table (terminal escape codes in the list file → mis-named results; rerun; `arith_probe.sbatch` now rejects such names). R2/R3/R4-S2-math submitted. Mixed corpus → next meeting.*

*Revision 60 (2026-10-18) — **GSM8K, the FineMath arithmetic curve, the R1-S1-math tail; FineMath stage 2 on R2–R4 decided.** (1) **GSM8K strict: R1-S2-math 0.2092 (39 % of the teacher), R1-S1-math 0.1524 (28 %)** vs R1-S2 0.0576 — 3.6×, z ≈ 12; flex/strict 1.00. With arithmetic 77 %, letter MMLU 60 %, HumanEval 49 %, MBPP 44 %: the FineWeb-era collapse of generative skills was mostly the corpus. (2) **Arithmetic curve on R1-S2-math's 9 checkpoints:** R1 38.3 → 52.6 at 100 M → ~59–61 at 200–300 M → 68 at 500 M → 76.9 at 1 B (≈ from per-condition values); FineWeb stage 2 at the same points 38.1 / 39.8 / 46.4. A fast part (+14 in 100 M; 2-digit and mul_2x1 at the teacher) and a slow part (+2.2 / 100 M after 300 M, ~3× FineWeb's +0.8; 4-digit only from 200 M), not saturated at 1 B. (3) **R1-S1-math's tail is real and model-specific** (`heldout_kl_seqs.py`): KL median maths 0.480 vs mean 1.437; its 6 web / 14 maths bad sequences are fine for R1, R1-S2 and R1-S2-math; ordinary text. **`ce_outliers`: not R5's mechanism** — no massive activation, weight drift ≤ 1.23× (R5 4–6×), `dt_bias` unchanged, kernel-independent; worst sequence CE 7.96; without the worst five +0.159 over R1 (plain forgetting of web). R5 on the two-corpus table: maths KL 2.06, 31 / 64 bad. (4) **Layout, paired over sequences:** KDA hybrid beats all-linear on maths by z +4.7 / +5.0 (stage 1 / 2), all-linear slightly ahead on web (z −2.4 / −3.2). (5) Decided: **FineMath stage 2 on R2, R3, R4 from their FineWeb stage 1** (`r{2,3,4}_stage2_*_finemath.yaml`, generic `stage2_math.sbatch`, only the final checkpoint kept). `heldout_kl.py` v2 adds an optional `LAYOUT=docs` (one document per sequence from its start, = DistillKit's format) — the default table is unchanged.*

*Revision 59 (2026-10-18) — **FineMath code results and the two-corpus held-out table.** (1) **Code:** HumanEval R1-S1-math **0.122**, R1-S2-math **0.122** (49 % of the teacher's 0.250; FineWeb students 0.012–0.055); MBPP 0.114 / **0.146** (44 %; FineWeb ≤ 0.062). Third skill — after arithmetic and the lettered-MC format — that the corpus, not the architecture, limited; already in stage 1 with the MLPs frozen. (2) **`heldout_kl.py`** (64 × 2048 per corpus, same sequences for all 12 models, validity OK): the FineWeb students' maths KL is **2.8–3.5× their web KL**; FineWeb stage 2 lowers both by the same fraction (−18 %; R2 −11 / −8 %); KDA < GDN in all 8 pairings; layout sign corpus-dependent (KDA hybrid −14 % on maths, all-linear ahead on web on this sample); cosine +12 % on maths. **FineMath stage 2 trades web for maths**: maths KL 0.810 → 0.442 (CE gap 60 % closed), web KL 0.293 → 0.456 (+56 %) — far more forgetting than the benchmarks show (0-shot −3.6 rel., cloze flat). Clean per sequence. (3) ⚠️ **R1-S1-math is heavy-tailed**: web KL 1.009, maths 1.437 (worse than R1's 0.993 on its own training domain) while the median CE gap on maths is only +0.140; 6 / 14 sequences > +0.5 nat, worst +5.4 / +4.4 — R5's signature (rev 37). Its DistillKit `eval_loss` 0.4392 is the stage-1 composite (0.2 CE + 0.2 KL + 0.6 hs_cos) on 200 other documents, so the logs could not show it. Owed: `heldout_kl_seqs.py` (median / trimmed KL, which sequences, same ones as R5?) and `ce_outliers` with `R5=<R1-S1-math>`. Until then rev 56's "maths alignment transfers worse" for its 0-shot drop is one of two explanations, and run B's start is suspect.*

*Revision 58 (2026-10-18) — **Exact FineMath arithmetic (`arith_stage2.py` v2).** R1-S1-math **68.5** (+30.3 [+28.3, +32.2] vs R1, G = 0.57); R1-S2-math **77.0** (+30.6 [+28.7, +32.6] vs R1-S2 — same start, only the stage-2 corpus differs; G = 0.69 of the remaining gap; +38.7 vs R1, G = 0.73). The corpus effect is the same size in stage 1 (MLPs frozen) as in stage 2 (everything trainable): **what the converted layers can do arithmetically depends on what they are trained on, not on unfreezing** — this also retracts rev 37's aside that the procedure "most plausibly lives in the MLPs". mul_2 is the exception (S1-math 50.0 > S2-math 43.5). Run B (FineMath stage 1 → FineMath stage 2) submitted 2026-10-18. New `heldout_kl.py`: CE + KL vs teacher on FineWeb AND FineMath held-out for every model, with ce_outliers' per-sequence check built in — no separate ce_outliers run for the FineMath models needed.*

*Revision 57 (2026-10-18) — ⭐⭐⭐ **FineMath restores arithmetic.** Probe v2 (pooled ≈ from per-condition values): R1-S1-math ≈ **68.6**, R1-S2-math ≈ **77.0** vs R1 38.3, R1-S2 46.4, teacher 91.0. The 4-digit cliff breaks (R1-S2-math add_4 37, sub_4 51; every FineWeb model 0–2.5), 3-digit nearly at the teacher (sub_3 88 vs 98.5). **This overturns rev 35's reading** ("the conversion preserved memorised facts and destroyed the general algorithm"): the procedure is recoverable by the same architecture with maths text, even with the MLPs frozen (stage 1). What limited arithmetic was the corpus. ⭐ FineMath stage 2 also restores the lettered-MC skill **across all subjects** (`mmlu_letters.py`: +13.3, p(gold) 0.285 → 0.416, 53 / 57 subjects up, largest in management / marketing / history), while cloze knowledge stays put (0.4052 → 0.4099). FineMath stage 1: arithmetic up but 0-shot down (68 % rel.) and cloze 0.3856 — the run-B criterion (R1-S1-math ≫ R1 on arithmetic) is met. Both logs clean: stage 1 tokenized in-job (877 progress lines, no NCCL timeout), stage 2 cache hit, max grad-norm 17.9 / 5.5, eval curves monotone on `finemath_4plus_heldout` (stage 2 KL 0.468 → 0.303) — NOT comparable with FineWeb-evaluated runs; `summarize_run.py` v3.2 says so. Batch-vs-single 7 / 8 on R1-S2-math (benign; teacher showed the same, rev 36).*

*Revision 56 (2026-10-18) — ⭐⭐ **FineMath stage 2 lifts letter-format MMLU to 0.4906 (60 % rel.)** — from R1-S2's 0.3581 (27 %), same start (R1), recipe and token budget, only the corpus changed. 0-shot stays at R1's level (mean 82.7 % vs R1 82.6 %, R1-S2 86.3 %): little forgetting. With rev 53–55 (knowledge in cloze 76–80 % for every student; the loss is the lettered-MC skill), this identifies the **corpus as the lever for the format skill**: FineMath's educational maths pages plausibly contain lettered questions and worked answers, FineWeb-Edu almost none. FineMath is 13-gram decontaminated against MMLU; cloze MMLU and `mmlu_letters.py` on R1-S2-math are still owed (does knowledge also rise, or only the format?). **FineMath stage 1 is worse** than R1 on 0-shot (68.4 % vs 82.6 % mean rel.), MMLU equal — aligning the linear layers on maths text transfers worse. **Generative evals (all queued jobs done):** GSM8K strict R1 seeds 0.024 / 0.036 / 0.032 (SD 0.006) → R1-S2 0.058 (11 % rel.; 300 M 0.031, 500 M 0.036), cosine 0.060, R2-S2 0.038, R3-S2 0.027 (flex/strict 2.4), R4-S2 0.001 (flex/strict 28). HumanEval (teacher 0.250): stage 1 all 0.012 (2 / 164), stage 2 0.030–0.055; MBPP (teacher 0.332): R1 0.046 → 0.060, all-linear ≈ 0. Code and GSM8K remain at 0–22 % of the teacher everywhere. Tables: `results.md` §7.*

*Revision 55 (2026-10-16) — **Matched-shot MMLU controls.** (1) Letter MMLU **0-shot**: teacher 0.6330, R1 0.2896, R1-S2 0.3428, R3 0.2789. With shots equal, the teacher still scores +19.0 above its cloze score and the students 6–12 below theirs (5-shot: +20.7 / −5 to −13) — caveat 1 of rev 54 is answered: the letter-format loss is not a shot-count artefact. (2) Cloze **5-shot**: teacher 0.4877 (+4.5 over 0-shot), R1 0.4125 (+1.5) → R1 retains 68 % at 5-shot vs 76 % at 0-shot: the teacher uses in-context examples about three times better (the same direction as the letter-format and GSM8K findings). Report retention as a range, 68–76 % for R1. (3) `mmlu_letters.py`, cloze 0-shot, teacher vs R1-S2: identical answer on 82.6 % of items; R1-S2 right-only 494 vs teacher right-only 1,027; 51 / 57 subjects lower, the largest drops in elementary (−20) and high-school (−12) mathematics — the knowledge gap sits in maths, consistent with the arithmetic probe. Both models over-pick the first option (~34 %): a shared answer-position/plausibility prior, which is part of caveat 2.*

*Revision 54 (2026-10-16) — **GDN students in cloze MMLU, and the two caveats on rev 53.** R2 0.3977 · R4 0.3938 · R2-S2 0.4021 · R4-S2 0.3969: all eight students sit at 0.394–0.405 (74–80 % of the teacher's 0.4432), the mechanism effect is ≤ 0.7 points (letter format and arithmetic: large), the layout ≤ 0.5 points, stage 2 +0.3 to +0.8. Natural mechanistic reading: every student carries the teacher's MLPs and embeddings byte for byte from stage 1 — where RADLADS locates factual knowledge — so knowledge that cloze can reach is equal across students; mechanism and layout act on the token-mixing skills (lettered MC, arithmetic, GSM8K format). ⚠️ **Caveat 1 (shots):** letter MMLU was 5-shot, cloze 0-shot, so the teacher's +20.7 letter advantage and the students' −5 to −13 mix format with shot count; the student-vs-teacher comparison *within* cloze is unaffected. Matched control: letter MMLU 0-shot on teacher, R1, R1-S2, R3 (rev 26 showed students gain from shots on ARC, so 0-shot letter should if anything lower them further). ⚠️ **Caveat 2 (sensitivity):** cloze gives the teacher only 19 points above chance (letter 40) and partly scores answer-string plausibility, which the students inherit with the teacher's embeddings; "knowledge retained" is therefore bounded to what cloze measures. Against "cloze just doesn't work for these models": it separates teacher from students by 6–8 SE and puts them in the same band as the other knowledge tasks, and cloze is the standard MMLU format *for* base and small models (it is the letter format that needs a learned skill).*

*Revision 53 (2026-10-16) — ⭐⭐ **Teacher cloze MMLU = 0.4432: the students retain 76–80 % of the teacher's MMLU knowledge.** Relative (chance 0.25): R1 76.4 %, R3 77.2 %, R1-S2 80.3 %, R3-S2 79.4 % — inside the 74–90 % band of ARC-C / HellaSwag / WinoGrande, not 6–27 %. The teacher scores 20.7 points HIGHER in letter format (0.6500) than in cloze; every student scores 5–13 points LOWER. **The MMLU "collapse" (rev 20: 13.9 %) is the loss of the teacher's lettered-multiple-choice skill, not of knowledge** — and the layout and stage-2 effects on MMLU live entirely in that skill. This closes the open MMLU question from rev 20 / 27 / 49–50 and replaces "MMLU 13.9 %" as a retention headline with "knowledge 76–80 %, format skill lost". STEM retains least (58–64 %), humanities/social/other 74–90 % — consistent with the lost arithmetic procedure. Not tested: GDN students in cloze (optional).*

*Revision 52 (2026-10-16) — ⭐ **MMLU in cloze format (`mmlu_continuation`, 0-shot): the four students are indistinguishable.** R1 0.3976, R1-S2 0.4052, R3 0.3991, R3-S2 0.4034 (±0.004) — against letter-format 0.3055 / 0.3581 / 0.2799 / 0.2750. In cloze, the layout effect is −0.2 / +0.2 points (letter: +2.6 / +8.3), and stage 2 adds only +0.8 (R1) / +0.4 (R3) (letter: +5.3 / −0.5). **So the students hold about the same retrievable knowledge whatever the layout; what separates them on MMLU is the letter format** — reading four labelled options and emitting the label of the right one, an in-context mapping across the prompt that the attention layers support and that stage 2 improved only in the hybrids. This replaces rev 49–50's reading ("stage 2 lifts MMLU") with "stage 2 lifts letter-format MMLU". ⚠️ Without the teacher's cloze score the retention figure is unknown — it is the one missing number (one debug job). Category pattern identical in all four (other > social > STEM > humanities).*

*Revision 51 (2026-10-16) — **MMLU letter check (`mmlu_letters.py`, R1 vs R1-S2, all 14,042 items): the stage-2 gain is real, and both models are dominated by letter bias.** Removing each model's letter prior leaves the gain intact (+5.26 → +5.95); p(gold) 0.265 → 0.285; McNemar z = +12.2; 44 / 57 subjects up. But R1 almost never answers B (6 % vs 24.7 % gold) and over-answers D (38.7 %); R1-S2 over-answers C (36.5 %); 45 % of answers change between the two, and p(gold) sits barely above chance in both. MMLU in lm-eval scores ONE letter token after options listed in the prompt — unlike ARC / HellaSwag / WinoGrande, which score the full answer text — so it measures knowledge *and* the in-context mapping of options to letters. Proposed decisive check: `mmlu_continuation` (same questions, full-text scoring) on teacher, R1, R1-S2, R3-S2 — if students recover most of the teacher there, the MMLU collapse is the letter format, not lost knowledge.*

*Revision 50 (2026-10-16) — ⚠️ **Rev 49 over-read the MMLU result; corrected.** (1) **Exact stage-1 MMLU raw for R2–R4** (only relative scores had been recorded before): R2 **0.2959**, R3 **0.2799**, R4 **0.2730** (the back-computed values were within 0.0002). (2) What stands: constant-LR stage 2 lifts MMLU in the hybrids (R1 +5.3, R2 +3.1 raw), not in the all-linear students. (3) What does not: *"the frozen MLPs were a large part of the collapse"* — stage 2 differs from R5 in **four** things at once (MLPs/embeddings/norms trainable, KL-only instead of 60 % hs_cosine, LR 1e-5 instead of 1e-3, +1 B fresh tokens), so the component is not identified; and the cosine variant of the same stage lifts R1 only to 0.3146, so the gain is recipe-sensitive. *"The recovered part needs attention layers"* holds only for our recipe: **Arcee's KDA-only reaches MMLU 55.8 (81 % rel.), their hybrid 55.1 (79 %)** — ours 27 % and 6 %. MMLU is **not explained**: most of the gap remains, and corpus, their ~1 B-token 32k fine-tune and batch size are all uncontrolled. (4) ⚠️ Near chance (0.25), MMLU accuracy can move with answer-letter bias; a `--log_samples` MMLU run (letter distribution of predictions) would separate calibration from knowledge.*

*Revision 49 (2026-10-16) — **0-shot and MMLU on the stage-2 finals and the seeds: stage 2 moves MMLU, but only where attention layers remain.** (1) **MMLU, stage 1 → 2:** R1 0.3055 → **0.3581** (13.9 → **27.0 %** rel., z ≈ 9.4), R2 ≈ 0.296 → 0.3268 (11.5 → 19.2 %), R3 0.2799 → 0.2750, R4 0.2732 → 0.2705 (flat). The first lever in the project that moves MMLU — R5's 7.6× frozen-MLP tokens did not — so the frozen stage was a large part of the MMLU collapse (rev 19 ranked it the weaker explanation), and what full-parameter KL recovers needs the nine attention layers: the layout gap on MMLU widens from 2.3–2.6 to **5.6–8.3 points**. Fits the retrieval reading of rev 28 (5-shot exemplars, letter mapping). (2) **0-shot rises slightly in all four** (mean rel. R1 82.6 → 86.4, R2 77.8 → 81.1, R3 83.6 → 84.8, R4 76.2 → 79.3; ≤ 1.3 raw points per task): no knowledge lost — the pre-registered non-degradation check holds. (3) **Cosine 5e-5: MMLU 0.3146 vs 0.3581 (−4.4, z ≈ −7.8)** — together with KL (+8 %) and CE, constant 1e-5 is settled; the run that first moved away from stage 1 lost MMLU, not 0-shot. (4) **Seeds:** 0-shot KDA − GDN +4.8 / +6.9 / +6.2 = **+6.0 ± 1.1** (robust); ARC-C seed SD 0.009–0.015 ≈ its sampling SE; **MMLU seed SD 0.005–0.007 > lm-eval's ±0.004**, and stage-1 KDA − GDN on MMLU (+0.009 / +0.023 / +0.005) is not robust. Single-run MMLU differences under ~2 points are noise; the stage-2 effects above are 4–7 SD. ⚠️ Stage-2 models were evaluated from the top-level directory (final weights), the queued GSM8K/code jobs from `checkpoint-15260` — same weights by construction; one `cmp` confirms. Tables: `results.md` §7.*

*Revision 48 (2026-10-16) — **`ce_outliers` on the five stage-2 finals: all clean.** Validity reproduced (teacher 2.2963, R1 2.4559), sdpa = FA2 to 1e-4, 0 non-finite logits, **no sequence more than +0.5 nat above R1** (worst ≤ 3.20) — none of them carries R5's latent damage, including the cosine run that first moved away from stage 1. Held-out web CE, stage 1 → 2 on the same 64 sequences (`test_r5_kl.py` → `ce_outliers`): R1 2.4559 → **2.4310** (gap to teacher closed 16 %), cosine → 2.4432 (8 %), R2 2.5082 → 2.5021 (3 %), R3 2.4379 → **2.4197** (13 %), R4 2.4623 → 2.4427 (12 %) — stage 2 moves CE far less than KL (−18 to −20 %), and even on its own training distribution most of the gap remains. R1-S2's gain grows slightly with position (−0.020 → −0.035); nothing accumulates. Weight drift vs R1 ≤ 1.03× (constant) / 1.07× (cosine), mostly short-conv kernels. R1 stage 1's large early-layer residual activations (layers 3–5) are gone in every stage-2 model — no teacher row, so not yet interpretable. ⚠️ The script compares everything with R1 stage 1: its per-sequence, position and drift sections for R2–R4 are cross-architecture. Details: `results.md` §7.*

*Revision 47 (2026-10-16) — **Arithmetic probe v2 on the stage-2 models, R1-S2's ten checkpoints and the four seeds.** (1) **Pre-registration met, but small:** R1-S2 pooled 38.3 → **46.4 %** (+8.1 [+6.5, +9.6], ≥ 43.3 % required); gap closure only **0.15** (teacher 91.0). R2 +3.7, R3 +5.7, R4 +6.4. Copy stays 100 % everywhere. (2) **Gains sit where the students already had partial ability** — add_2 77 → 89.5, sub_2 55 → 80, add_3 9.5 → 20.5, sub_3 3 → 17.5 — while **4-digit stays at 0–2.5 %** (teacher 79–80) and mul_2 barely moves (18 → 20): the problem-space cliff moves a little, the general procedure is still missing. (3) **Seed noise measured for the first time:** pooled SD 1.1 (R1) / 1.7 (R2) points across data orders; KDA − GDN (hybrid, stage 1) +13.6 / +11.5 / +11.1 = **+12.1 ± 1.3** — the mechanism effect is ~10× the seed SD. ⚠️ The paired bootstrap CIs cover problem sampling only; the SD of a difference between two single runs is ≈ 2 points, so **effects under ~4 points are not established** (R2-S2's +3.7, KDA layout −1.9, cosine +2.6). (4) **Mechanism gap does not close in stage 2:** KDA − GDN hybrid +17.9 (was +13.6), all-linear +9.3 (was +10.0). (5) Layout within stage 2: KDA −1.9 (noise), **GDN +6.7 for all-linear** — opposite to held-out KL, one seed; open. (6) **Cosine 5e-5 vs constant:** +2.6 on arithmetic (49.0 vs 46.4) against −8 % on held-out KL — no clear winner on behaviour; wait for GSM8K/MMLU/ce_outliers. (7) **Token curve:** flat for the first ~300 M (38.1 / 38.8 / 39.8), then rising ~**+0.8 points per 100 M** with ±1.2 checkpoint noise (= seed noise), no plateau at 1 B; the budget-matched 300 M point is +1.5 only. (8) **R1-S2 ≈ R5 on arithmetic** (46.4 vs 45.9) — 1 B of full-parameter KL reaches what 2.29 B of frozen-MLP stage 1 reached, without R5's damage — and **neither closes more than ~15 % of the gap.** Unfreezing the MLPs was not the missing lever on web text; the data is the remaining candidate → **the FineMath stage-2 run is indicated** (thesis-plan rule: "if FineWeb stage 2 does not recover arithmetic, the math run is the test of the corpus hypothesis"). Its cleanest readout: the arithmetic slope per 100 M on FineMath vs the +0.8 measured here.*

*Revision 46 (2026-10-16) — **Queue vs debug split final; stage-2 data practice recorded for the methods chapter.** (1) `queue_evals.sh` v4: `all` = GSM8K + HumanEval + MBPP on the 5 stage-2 finals, GSM8K on R1-S2 ck 4578/7630 and the 4 seeds, HumanEval + MBPP on teacher + R1–R4 (31 lprod jobs); `debug` submits nothing — it writes the arithmetic model lists and prints the arithmetic-probe and `ce_outliers` sbatch lines to run by hand, two at a time. (2) **Why the stage-2 split is standard practice, for the thesis:** LLM pretraining and distillation pipelines train for (at most) one pass over fresh data and avoid repeating documents across stages unless the corpus is too small; RADLADS draws all its steps from one large source (DCLM) at well under 1 % of it, which in practice means unseen documents in each step. Continuing the same seed-42 permutation of FineWeb-Edu past the stage-1 pool is exactly "keep streaming the corpus", and the exact-duplicate skipping and the held-out exclusion are the deduplication / decontamination steps such pipelines apply. What we did NOT do (and say so): cross-snapshot near-duplicate removal (MinHash) — FineWeb-Edu only dedups within each crawl; and the stage-1 pool itself was not decontaminated against held-out (9 / 2,000 exact leaks, rev 42).*

*Revision 45 (2026-10-16) — **Eval logistics fixed after testing them against the cluster's rules.** (1) **The debug QoS cancels every job beyond two per user** (not pending, cancelled), so nothing is queued there any more: the 0-shot trio and MMLU are run by hand on debug; `queue_evals.sh` v3 sends GSM8K, HumanEval, MBPP, the arithmetic probe and `ce_outliers` to lprod (command-line `--qos/--time` override the scripts' debug headers). Walltimes generous: GSM8K 6 h, HumanEval 8 h, MBPP 12 h, arithmetic 3 h, outliers 2 h. (2) **MBPP's protocol confirmed from the installed task file:** `num_fewshot: 3`, `fewshot_config: first_n` from `list_fewshot_samples` (the 3 fixed prompt problems of Austin et al.), greedy, stop at `[DONE]`, metric pass@1 — our script passes no `--num_fewshot`, so the task default applies. (3) `eval_lm_afm.sbatch` back to **32 CPUs** (as every earlier eval; cost is not the constraint, and it removes one difference from the runs it is compared with). (4) **`finalize_checkpoint.py` is not needed for evaluation:** lm-eval and the probe pass `use_cache=True` explicitly, which is all its flip did for them. What *is* needed is that each checkpoint carries `modeling_afm_kda.py` and tokenizer files; `queue_evals.sh` now checks that on the login node before submitting, instead of failing after days in the queue. (5) **`arith_stage2.py`** added: `arith_compare.py` only knows the stage-1 names, so the stage-2/seed comparisons (stage 2 − stage 1 per student with the pre-registered ≥ R1 + 5 points, KDA − GDN and layout within stage 2, cosine − constant, the R1-S2 token curve, seed spread) read the same result files. The probe itself (`arith_probe.py` v2) is unchanged on purpose — same 2,286 problems as every earlier model. (6) **FineMath:** download 3.2 M docs (~4.6 B Llama tokens) so a run of up to ~2 B tokens fits; everything (raw, splits, DistillKit's tokenize cache) on `$FAST`. One training pool is enough for runs that start from the same model, because constant-LR stage 2 makes every shorter budget a checkpoint of the longest run and DistillKit's seed-42 shuffle makes a shorter run's data a prefix of a longer one's; disjoint pools (`--pool-a-docs`) only for two runs stacked on the same model.*

*Revision 44 (2026-10-16) — **All stage-2, seed and LR-check logs read: everything clean; the mechanism and layout gaps survive stage 2; data-order variance is ~20× smaller than the mechanism gap; HALO's cosine 5e-5 loses.** (1) **Stage 2, held-out KL (T = 2, `eval_loss`, same 200 docs), 100 M → 1 B:** R1 0.3379 → **0.2731** · R2 0.3726 → **0.2965** · R3 0.3563 → **0.2937** · R4 0.4022 → **0.3298**; monotone in all four, **still falling** at 1 B (−0.005 to −0.009 per 100 M). GDN/KDA = **1.086** (hybrid) and **1.123** (all-linear), barely narrower than at 100 M (1.103 / 1.129). All-linear/hybrid = **1.075** (KDA) and **1.112** (GDN) at every eval point — the hybrid is better, in the same direction as the stage-1 `eval_loss` on these same documents (R3 0.5744 vs R1 0.5686; the sign flip of rev 29 was against the *other* held-out sample). R3 ≈ R2 now: mechanism and layout cost about the same. (2) **Seeds (data order only):** final held-out composite R1 0.5686 / 0.5697 / 0.5693 (SD 0.0006), R2 0.5814 / 0.5814 / 0.5813 (SD 0.00006); the paired gap R2 − R1 is 0.0128 / 0.0117 / 0.0120 = **0.0122 ± 0.0006** (t ≈ 37). Same seed = same batches for R1 and R2, so the pairing is exact. ⚠️ Measures data-order variance within one 500 k pool with fixed initialisation — a lower bound on full seed variance. ⚠️ The train-stream KL (last 20 rank-0 microbatches) spreads 0.283–0.314 across R1's seeds — as wide as the R1–R2 gap: it cannot rank runs (rev 25's trap, now quantified). All four seed runs passed every health check; first microbatches differ from R1/R2 (seed took effect); throughput reproduces (R1 25.5–25.6 k, R2 29.7–29.9 k tok/s). (3) **LR check: cosine 5e-5 → 1e-5 is WORSE.** KL 0.528 at 100 M (constant: 0.338) — the 5× higher LR first undoes much of stage 1 — then recovers as the LR decays, ending **0.2956 vs 0.2731 (+8 %)** at equal tokens. Its slope at the end is steeper (−0.014 per 100 M), so it might catch up with more tokens, but at the 1 B budget constant 1e-5 wins. HALO's value was tuned after a *layer-local* stage 1; ours already trained end to end. Recipe of record stays constant 1e-5; `ce_outliers` on the cosine model is queued because a run that first moved away from its start is exactly R5's risk pattern. (4) **Grad-norm spikes (max 8–19 per stage-2 run) are harmless:** `grad_norm` is the global L2 norm of the gradient over all trainable parameters, logged BEFORE clipping; `max_grad_norm: 0.5` rescales the applied gradient, not the logged number. In stage 2 the typical norm is ~1.4, so *every* step is clipped (factor ~0.35); a 15.95 step was scaled by 0.031. Only every 10th step's norm is logged. No eval rise anywhere; R1-S2's spike at epoch 0.9438 was followed by eval 10 still improving (−0.0056). Contrast R5: 83,070 at LR ~8e-4 with eval collapse. Most likely cause: an outlier sequence somewhere in the global batch (rank 0 shows a 0.64-KL microbatch in that step, but sees only 2 of 32 sequences). (5) **`summarize_run.py` v3.1:** tokens per eval from the epoch (v3.0 assumed eval_steps = max_steps/10 — wrong for stage 1, where eval_steps is 500 and the 10th eval is the end-of-run one), spikes measured against a running median after warm-up (v3.0 counted stage-1 warm-up gradients as 14–18 "spikes"), `--compare` adds `loss last20` (all ranks, 200 steps — the stable train-side number). (6) **Eval scripts v2:** 0-shot removed (run by hand on debug), MMLU to the debug queue, HumanEval/MBPP without `--num_fewshot` (task defaults: 0 and 3 fixed prompts), arithmetic lists via file. (7) **FineMath prepared like FineWeb-Edu:** `get_finemath.py` (download on the serial partition: pinned revision, seeded shard order, footer row counts, AFM calibration; prepare: text only, 2,000 held-out carved first, exact-text leaks removed, train written in source order). Raw parquet on `$FAST`. (8) **Storage:** work at 1.8 / 2 TB; stage-2 outputs ≈ 110 GB each (10 weights-only checkpoints + top level).*

*Revision 43 (2026-10-15) — **R5's MMLU and GSM8K land; stage 2, seeds and the LR check are all finished; first stage-2 log read; meeting decisions.** (1) **R5 MMLU 0.3068 (14.2 % rel.) = R1's 0.3055 (13.9 %)**, Δ +0.0013, z ≈ 0.2, 95 % CI on Δ ≈ [−0.010, +0.012]: **7.6× the stage-1 tokens bought nothing on MMLU**, so with the MLPs frozen the token *budget* is not what holds MMLU down — the MMLU explanation narrows to *no unfrozen stage* and/or *the corpus* (stage 2 tests the first). **R5 GSM8K strict 0.0409 (7.6 % rel.) vs R1 0.0243**, Δ +0.0166, z ≈ 2.4 — a small real gain, in line with R5's +7.6-point arithmetic gain (heuristic check: arithmetic ratio 45.9/38.3 = 1.20, 1.20^2.28 = 1.51 → predicts 0.037 vs 0.041 measured). R5's flexible/strict ratio 1.09 (R1 1.44) — it keeps the answer format. Table in "The 0-shot benchmarks — and they did not move" below. (2) **R1-S2 (59749921) clean:** 11.83 h, 23,491 tok/s, 1.0005 B tokens measured (488,507 sequences); held-out `eval_loss` — **KL only (T = 2), on the same 200 held-out documents** — falls monotonically **0.3379 → 0.2731** over ten evals (−19 %), still falling at the end (last steps −0.006 per 100 M); train-stream KL (`loss`) 0.344 → 0.276; max `grad_norm` 15.95 at epoch 0.944 (one late spike; mean of last 20 = 1.44, stage 2 runs above the 0.5 clip by design); no `Froze` line (correct: nothing frozen); memory 12.82 GiB. (3) ⚠️ **`summarize_run.py` v2 was wrong for stage 2** — it printed "final vs R1 −0.2955" by subtracting R1's stage-1 *composite* (0.2 CE + 0.2 KL + 0.6 hs_cos = 0.5686) from a stage-2 *KL* — a meaningless number; it expected "Froze 147 ×16"; and it hard-coded the stage-1 loss keys, so the stage-2 component section (`distillation_loss/1_kl`) silently vanished. The cluster copy was also older than the project copy (no spike checks). **v3** detects stage and components, uses stage-specific references, adds a seed check (first microbatch must differ from R1/R2) and `--compare` for side-by-side curves. (4) **Meeting with Anamaria:** one unified thesis (no Part I / Part II split; the practical work contributes only chosen experiments), five chapters, RQs fixed only after the experiments, inference analysis = forward-pass runtime vs sequence length (KLA paper App. E.3), HumanEval added (base `humaneval`, not `humaneval_instruct`), possible math-specific distillation on FineMath, no fine-tuning, queue everything at once. Plan: `claude/thesis_plan.md` rev 3. (5) New scripts: `eval_lm_afm.sbatch` (all lm-eval regimes incl. HumanEval/MBPP, `--log_samples` always, 8 CPUs instead of 32 — a 32-core 1-GPU job is billed as a full node), `queue_evals.sh` (whole battery in one go), `arith_probe.sbatch` `SET=custom`.*

*Revision 42 (2026-10-12) — **Stage 2 launched on all four students, seed replicates and an LR check queued; resolved TODOs marked; document becomes the archive.** (1) **Zero-overlap split rebuilt (`make_stage2_split.py` v4).** FineWeb-Edu contains EXACT DUPLICATE texts (it deduplicates within each Common Crawl snapshot, not across them): the stage-1 pool of 500,000 holds 1,131 internal duplicates, and **9 of the 2,000 held-out texts also occur in stage 1** (eval contamination 0.45 %, R1–R5 alike). Walking the seed-42 permutation from position 500,000 and skipping every text already in stage 1 (6,290), in held-out (25) or already kept (8,544) gives **1,350,000 documents from positions 500,000–1,864,858, overlap 0 / 0**, recorded in `provenance.json`. This corrects rev 41's positions; `stage2_prep.sbatch` was never used and is deleted. (2) **Warm-up done by hand on the debug queue** (single process): *Adding EOS* 3:46 + *Tokenizing* 3:28 + *Packing* 0:01 for 1.35 M documents ≈ 7.5 min — it would have blown the 600 s barrier inside a 4-node job. 49 `cache-*.arrow` files beside the split, shared by all four students (the cache key is data + teacher tokenizer + `num_proc`, not the student). (3) **Stage 2 for R1–R4** (jobs 59749921 / 59749924 / 59749930 / 59749932): identical recipe — KL 1.0 at T = 2, nothing frozen, constant 1e-5 after 380 warmup steps, 15,260 steps = 1.0 B tokens, weights-only checkpoints every 1,526 steps (`save_only_model`: ~10 GB each instead of ~60 GB; cost: a crash restarts with fresh Adam). (4) **Seed replicates** of R1 and R2 stage 1 with `training_args.seed` 43 and 44 (`make_seed_replicates.py`): data ORDER only — `dataset.seed: 42` keeps the same 500 k pool, the same 200 eval documents and the same cache; the new parameters were drawn once at build time, so this measures data-order variance, not initialisation variance. (5) **R1 stage-2 LR check queued 2026-10-12**: cosine 5e-5 → 1e-5 (`r1_stage2_kda_hybrid31_cosine5e5.yaml`), HALO's stage-2 value for a Qwen3-4B-shaped model (same 36 × 2560 shape as AFM). (6) **HALO (Chen et al. 2026) read** — the closest related work: same corpus (FineWeb-Edu), 25 % attention, 320 M-token stage 1, 1 B-token KL stage 2. It **corroborates R5** (more stage-1 tokens do not help; 1.3 B was worse on retrieval than 320 M), **questions constant 1e-5** in stage 2 (hence the LR check), and reports that **their KDA conversion diverged** in stage 2. (7) **Disk cleanup 2026-10-08**: deleted `output/resume_test`, `output/r1_stage2_smoke`, `distillkit/output/smoke` and **R5's checkpoints 3494–31446** (their probe results stay in `eval_results/arith_probe_v2`; ⚠️ R5's curve can no longer be re-evaluated). (8) **Decisions:** stage-1 recipe final at 300 M; **cut**: FLP layout, de-novo ablation, corpus-mix run, IFEval (base teacher, no chat template), long-context fine-tune; **planned**: retrieval probe (needle + key-value recall), inference speed (prefill, decode, max batch), HumanEval + MBPP. (9) Every previously open item that is now settled is marked ✅ in place; the index is under "How to read this document".*


*Revision 41 (2026-10-08) — **Stage 2 gets a ZERO-overlap training split.** `make_stage2_split.py` reproduces DistillKit's exact `shuffle(seed=42)` of the train split, excludes positions 0–499,999 (the selection R1–R4 drew from; R1 consumed ~383 k of them), and saves positions **500,000–1,849,999 (1,350,000 docs, ~1.06 B tokens)** as `datasets/fineweb_edu_3B_train_stage2_nooverlap`, after **verifying by document id** that it shares nothing with stage 1 or the held-out split (`provenance.json` records the counts). `stage2_prep.sbatch` builds it, runs one single-process DistillKit warm-up so all preprocessing happens without the 600 s barrier, then repeats the run and prints **CACHE HIT** only if nothing is re-tokenized. The stage-2 YAML points at the new split; `seed: 42` still selects R1's 200 eval documents. Corrects rev 39's claim that RADLADS reused documents — it reused a *source*. ⚠️ *(Corrected in rev 42: `stage2_prep.sbatch` was never used and is deleted; the split had to skip exact duplicate texts and spans positions 500,000–1,864,858.)**

*Revision 40 (2026-10-08) — ✅ **Stage-2 smoke (59735584) passed.** Guards OK; starts from R1 `checkpoint-4580` (`global_step 4580`, 4,999,026,076 params, `modeling_afm_kda.py` sha1 7291d97 = R5's); **no `Froze` line** (nothing frozen, as intended); first KL microbatches 0.18–0.57, i.e. **R1's level — the right weights loaded**; `eval_loss` (KL, T = 2) 0.3775; `grad_norm` 0.98 → 0.84 (above the 0.5 clip — expected with embeddings, MLPs and `lm_head` now trainable; watch it); `train_mem_gpu_alloc_delta` **13.8 GB** (R1 10.7), peaked +8.4 GB — fits easily; weights-only checkpoint at step 10 written in ~10 s (`save_only_model` works under ZeRO-2); steady **~2.8 s/step ≈ 23 k tok/s → ~12 h** for 15,260 steps. ⚠️ **Data changed back to seed 42, full split**: the smoke run's fresh seed-43 draw spent ~26 s in TRL's *Adding EOS* pass at ~900 docs/s, which would be ~25 min for 1.35 M docs — a certain 600 s barrier timeout. Seed 42 over the full split is exactly R5's already-cached dataset (instant), and RADLADS uses one dataset for all steps anyway. `dataset_num_proc` must stay 16 (cache key). Framing for the thesis: **Arcee's merged stage 1 (RADLADS 1+2, MLPs frozen) followed by RADLADS step 2 proper** — neither pipeline exactly, a deliberate combination.*

*Revision 39 (2026-10-08) — **Stage 2 now follows RADLADS step 2 as published, and cites it instead of choosing.** Re-reading both sources: Arcee *"collapsed the first two distillation stages into a single one with frozen MLP parameters … then allowed the long context fine tune to pick up any slack in adjusting the MLP layers"* — so R1–R4 are RADLADS steps 1+2 merged, MLPs frozen. **RADLADS step 2 itself is a full-parameter KL distillation** (KL to the teacher's logits only, flat 1e-5, 250–700 M tokens; *"freezing the MLPs and embeddings … results in significantly reduced model performance"*). **RADLADS step 3 and Arcee's fine-tune are NOT distillation**: plain cross-entropy, no teacher, long context. So `r1_stage2_kda_hybrid31_unfreeze` is now **KL 1.0 only, nothing frozen, constant 1e-5, 1.0 B tokens, checkpoints every 100 M** — RADLADS step 2 on top of Arcee's merged stage 1, which is exactly the step that merge dropped. Deviations, each stated: seqlen 2048 (theirs 512), FineWeb-Edu (theirs DCLM), KL temperature 2.0 (theirs unstated; = our stage 1). ⚠️ *(Corrected in rev 41: RADLADS uses the same dataset **source**, DCLM, for all three steps — at < 1 B of ~4 T tokens that almost certainly means different documents. It is not evidence for reusing documents.)* Expect **no `Froze` line**.*

*Revision 38 (2026-10-08) — **Stage 2 redesigned before launch; supersedes revision 36's S2a.** Now `r1_stage2_kda_hybrid31_unfreeze`. Changes and why: **(1) LR 1e-5 CONSTANT after 380 warmup steps** instead of cosine — this is what RADLADS step 2 (flat 1e-5 / 7e-6) and the xLSTM-distillation stage II (constant 1e-5) use, Arcee's fine-tune LR is also 1e-5, and a constant LR makes every checkpoint a comparable point and the run extendable by resuming; it also retires the arbitrary `min_lr_rate: 0.1`. **(2) 1.0 B tokens** (15,260 steps), Arcee's budget, with **weights-only checkpoints every 100 M** (`save_only_model`, ~10 GB each instead of ~60 GB with all MLPs carrying Adam state) — `checkpoint-4578` is the 300 M point comparable to R1. **(3) A fresh draw of data**: seed 43, 1,350,000 documents (~280 s of rank-0 tokenizing, inside the 600 s barrier), ~13 % chance overlap with R1's documents. Unchanged from rev 36: start from R1 `checkpoint-4580`; MLPs unfrozen, embeddings / `lm_head` / norms frozen (**Froze 75 ×16**); CE 0.2 / KL 0.8 at T = 2, no `hs_cosine`. ⚠️ **What Arcee actually did is less specified than this document implied**: their blog says *"fine tuning at 32k context … about a billion tokens … 1e-5"* — it names neither the loss nor the frozen set, and "fine tuning" most plausibly means plain next-token CE on all parameters. Ours is a **distillation** stage (KL-dominant), at 2048, MLPs only: closer to RADLADS step 2 than to Arcee. ⚠️ `eval_loss` is now 0.2·CE + 0.8·KL on a different 200-document draw — compare it within this run only.*

*Revision 37 (2026-10-07) — ⭐⭐⭐ **The R5 contradiction is resolved: the divergence left PERMANENT, INPUT-TRIGGERED damage.** `ce_outliers.py`: **sdpa and FA2 agree to 0.001** (not the kernel). On web text **4 of 64 sequences are catastrophic** (+3.7 to +6.6 nats over R1); **without the worst five, R5 − R1 = +0.0002** — on ordinary sequences R5 *is* R1. Tulu 3: 9 of 64 catastrophic (worst +10.3). R1 has no outliers (worst sequence 3.18 vs teacher 3.05). The excess is flat across positions, so nothing accumulates along the sequence; a bad sequence is bad from the start. **On R5's worst sequence the residual stream carries a massive activation from layer 3 onward: max |h| 1,408 vs R1's 162, RMS 13–16× R1's at every layer to 35** — on ordinary English text ("…Passover is not a day…"). The frozen MLPs and norms downstream receive inputs far outside anything they were trained on. **Weight drift points at the source: layer 0's KDA projections are 4–6× R1's norm** (`v_proj` 5.74×, `f_proj` 5.68×, `q_proj` 5.64×, `o_proj` 4.96×), plus the attention layers' `v_proj`/`o_proj` at 19/23/27 (3.4–3.9×). **`dt_bias`, the decay gate, did not move** — this is not a state that stopped forgetting. 147 frozen tensors bit-identical. ⭐ **Consequences:** R5 is unusable as a base model; R1 stays. It also explains every R5 oddity at once — `eval_loss` near R1's, 0-shot slightly lower, Tulu 3 4.6× perplexity, arithmetic fine (short prompts rarely trigger it). ⭐⭐ **Thesis-worthy lesson: a run that diverges and "recovers" can carry latent damage that no mean metric shows** — only the per-sequence distribution does. `ce_outliers.py` (with `R5=<path>`) is now part of checking any new model, S2a included.*

*Revision 36 (2026-10-07) — ✅ **The batching control clears the probe.** At batch size 1, ~4.5 % of answers change (R1 2184/2286 identical, R5 2190/2286), almost all of them wrong answers changing into different wrong answers. Pooled accuracy moves +0.3 (R1) and +0.1 (R5), and **R5 − R1 goes from +7.6 to +7.4 points**. Every conclusion from revision 35 stands. ⚠️ `ce_outliers.py` printed nothing and exited 0 — almost certainly an **empty script file on the cluster** (Python runs an empty file silently); the R5 CE contradiction is still open. ⭐ **Decision: R1–R4 stay the stage-1 models.** R5 diverged, matched R1 on held-out loss and 0-shot, and bought +7.6 arithmetic points for 7.6× the tokens — **a longer stage 1 is not the lever.** ⭐⭐ **Stage 2a queued: R1 `checkpoint-4580` with the MLPs UNFROZEN, `lr 1e-5`, CE 0.2 / KL 0.8, no `hs_cosine`, same data and 4580 steps as R1** (`s2a_kda_hybrid31_unfreeze.yaml`, `s2a.sbatch`, behind a 20-step smoke). Pre-registered: **Froze 75 tensors ×16**; held-out curve monotone, max `grad_norm` < 20; **arithmetic pooled ≥ R1 + 5 points (≥ 43.3 %)** with `add_3` rising most; **0-shot within 1 point of R1** on all three; **GSM8K strict above R1's 2.43 %**. If arithmetic does not move, the MLPs were not the bottleneck and the data lever (S2-b: frozen MLPs + a few % synthetic arithmetic) is next.*

*Revision 35 (2026-10-07) — ⚠️⚠️⚠️ **R5 DIVERGED during training, repeatedly, and recovered.** Held-out `eval_loss` went 0.698 → 0.638 → **1.742** → 1.062 → **1.755** before a clean descent to **0.5715** — *worse* than R1's 0.5686, so pre-registration A **failed**. `grad_norm` peaked at **83,070**; at checkpoints 6988 and 10482 the model **could not copy a three-digit number** (0–2 %). Every spike happened while the LR was above ~5e-4, which R5 held for **17,580 steps against R1's 2,304**: **1e-3 is not a safe peak LR for a long run.** ⚠️ R5 is therefore not a clean "more tokens" experiment. ⚠️⚠️ **`test_r1_ce_math.py` on R5 contradicts DistillKit's own eval**: web gap +0.4669 (R1 +0.1595), Tulu 3 **+1.53**, with a paired spread **20× R1's** — a few catastrophic sequences, not uniform damage. Pre-registration B "failed" as measured but is **uninterpretable until `ce_outliers.py` explains it**. ⭐⭐⭐ **The arithmetic probe v2 ran on all five students, the teacher and R5's ten checkpoints.** Students copy perfectly (**P1 ✅** — the failure is computation, not reading). ⭐ **AFM's tokenizer makes every 1–3-digit number ONE token**, which **retracts revision 32's digit-alignment explanation** for those sizes — and the carry/no-carry split shows **no excess carry penalty (P3 ❌)**. What the students lose tracks the **size of the problem space**: ~80 % of the teacher on 2-digit tables (10³–10⁴ pairs), ~10 % on 3-digit (~10⁶), zero on 4-digit — **memorised facts survive, the general procedure does not.** Mechanism **KDA +13.6 / +10.0 points** over GDN, tight CIs (**P4 ✅**); layout **no effect** on arithmetic. **R5 − R1 = +7.6 points [+6.1, +9.2]** — **SMALL GAIN**, which **falsifies my pre-registered NULL**.*

*Revision 34 (2026-10-06) — ✅ **R5 finished: 34,940 steps, all ten checkpoints kept** (3494 … 34940, exactly as `save_steps × save_total_limit` was set up to give). ⚠️⚠️ **And 7.6× more FineWeb-Edu tokens did NOT improve a single 0-shot benchmark** — R5 is *below* R1 on all five metrics (ARC-C `acc_norm` 0.5205 → **0.5000**, WinoGrande 0.6630 → **0.6464**, HellaSwag 0.7240 → 0.7220), mean relative score **82.6 % → 77.8 %**. No single difference clears 1 σ unpaired, so "worse" is not established — **but "better" is ruled out**: the 95 % intervals cap any gain at about +2 points per benchmark. ⭐⭐ **Everything else about R5 is pre-registered here before it is measured**: held-out `eval_loss`, the web/maths/instruction CE gaps, and a new arithmetic probe (`arith_probe.py`) with a **copy control**, a **digit-length sweep** and a **carry/no-carry split**, scored mechanically by `arith_compare.py` against thresholds written below. ⚠️ Stated honestly: these predictions were written **after** R5's 0-shot results were seen, which correlate with them, and **before** any arithmetic, held-out or CE measurement of R5.*

*Revision 33 (2026-10-02) — ⚠️⚠️ **R5 died in dataset preprocessing 30 seconds short of finishing, and the cause is a 600-second NCCL barrier.** TRL's `_prepare_dataset` wraps the tokenize/pack pass in `accelerate.main_process_first()`: rank 0 tokenizes all **2,921,000** documents while ranks 1–15 wait in `barrier()`, whose watchdog aborts at **600 s**. Rank 0 sustained **4,819 ex/s → 606 s projected**. ⭐ **It needed 4,868 ex/s and delivered 4,819 — it missed by 1 %.** R1–R4 never hit it because `num_samples: 500000` made that work 106 s; **removing `num_samples` to get a full epoch is the change that killed it.** ✅ **Fixed by warming the `datasets` map cache in a single-process job** (no barrier to time out): 474 s alone on the node, cache written beside the dataset's arrow files, and a second run then went straight from loading the teacher to `Starting training.` with no tokenize pass at all. ⭐⭐⭐ **And that warm-up measured the epoch exactly**: 559,146 steps at batch 2 → **1,118,292 packed sequences = 2.2903 B tokens = 784.1 effective tokens/document**, against the 774 estimate — so `max_steps` for one epoch is **34,946.6**, and R5 runs at **34,940** (99.98 %) with `save_steps: 3494` and `eval_steps: 1747` dividing it exactly. **R5 submitted as job 59234295**, 4 nodes, `--time=48:00:00`.*

*Revision 32 (2026-09-29) — ⭐⭐⭐ **The GSM8K collapse is RESOLVED: it is the arithmetic circuit.** A single-step probe (`"What is 437 + 62?"`, no long context) gives **teacher 76.0 %, R1 20.0 %, R2 7.5 %, R3 24.0 %, R4 9.5 %**. Fitting the effective operation count on the teacher alone (`0.76^k = 0.5353` → k = 2.28) then predicts **R1 at 2.56 % against a measured 2.43 %** — the first account in this document to reproduce the number. ⭐⭐ **Two independent damages on two axes**: arithmetic/precision tracks the **mechanism** (KDA ~2.5× GDN — the largest mechanism effect in the project), format/termination tracks the **layout** (98 % → 56 → 45 → 31 → **1 %**). ⚠️⚠️ **Retracts revision 30's "degeneration, not arithmetic" and revision 31's multiplicative decomposition** — the latter conditioned on a post-treatment variable and is refuted outright by R4 (1 % termination, 8 % flexible). Degeneration is a **symptom**. ✅ Pre-registration: the `flexible/strict` format meter **confirmed exactly** (termination order is its precise inverse); the loop-rate proxy **falsified**.*

*Revision 31 (2026-09-29) — ⚠️⚠️ **The full GSM8K and IFEval runs wrote no sample files, and the generations are unrecoverable.** `eval_generative_afm.sbatch` welded `--log_samples` to `--limit`, so only smoke runs logged text; `results_*.json` holds aggregate metrics and nothing per-item. The ten generations that produced revision 30's finding survive only because they came from a smoke run. **Fix: decouple the flags and log by default** — samples are ~10 KB/item, 13 MB for all of GSM8K. **Re-run plan: 200 items × five models with `--log_samples`**, which measures a ~60 % loop rate to ±3.5 % for ~2 h of GPU, with the teacher's rate as the missing baseline.*

*Revision 30 (2026-09-29) — ⭐⭐⭐ **The GSM8K error analysis ran, and it overturns three revisions of explanation. The dominant failure is DEGENERATION, not arithmetic.** In ten consecutive R1 generations, **six collapse into verbatim repetition and never emit `####`**; one repeats a single sentence sixteen times without attempting a calculation. The arithmetic it does attempt is **60 % correct per operation** — mediocre, not broken. ⭐⭐ **And the two rates together finally reproduce the score**: `0.60⁴ × 0.40 = 0.052` against a measured **0.0243**, where arithmetic alone required an implausible per-step 0.395. ⭐⭐⭐ **This unifies the whole document**: repetition collapse is a long-range retrieval failure, which is why the `flexible/strict` ratio runs **1.44× → 40.75×** from hybrid to all-linear, why MMLU separates the layouts at 4.3–4.7 σ, and why nothing separates them on 0-shot tasks or held-out CE/KL. **The `flexible/strict` ratio was a repetition meter all along.** ⚠️ n = 10, and the teacher's own loop rate is unmeasured, so "60 %" has no baseline yet.*

*Revision 29 (2026-09-29) — ✅ **Held-out `hs_cosine` measured; the teacher validates all three components at zero.** All three favour KDA in both layouts, and **KL is the discriminator** (+22.1 % / +13.6 % for GDN, against CE's +2.1 % / +1.0 %). ⚠️⚠️ **But the LAYOUT effect on held-out loss flips sign between this project's two held-out samples**, for both mechanisms — so "all-linear is slightly better on held-out CE/KL" is withdrawn: **the layout is not reliably measurable on short-context language modelling at all.** That sharpens rather than weakens revision 28: no measurable layout difference on packed web text, **4.3–4.7 σ on MMLU**, 1.44× → 40.75× on GSM8K format compliance. ⚠️⚠️ Also adds an explicit admission: **the GSM8K explanation is inadequate.** The multiplicative-composition argument requires per-step accuracy to fall from **86 % to 40 %**, which a 17 % perplexity penalty does not produce; both recorded errors are **operand copies**, not arithmetic slips; and **single-step arithmetic has never been measured.** Two cheap experiments named.*

*Revision 28 (2026-09-29) — ⭐⭐⭐ **The 2×2 is complete on all five benchmarks plus held-out CE and KL, and the layout question resolves.** **Mechanism: KDA beats GDN seven measurements out of seven, in both layouts** — held-out KL puts GDN +22.1 % (hybrid) and +13.6 % (all-linear). **Layout: free on short tasks, expensive on long ones.** All-linear is *better* on held-out CE, held-out KL and the 0-shot mean — and **worse on MMLU by 4.7 σ (KDA) and 4.3 σ (GDN)**, the only 5-shot benchmarks. ⭐⭐ Corroborated independently by GSM8K's `flexible/strict` ratio, which runs **1.44× → 1.73× → 3.93× → 40.75×** in exactly the layout order: `gdn_only` emits the required `####` marker **once in 1,319 problems**. ⚠️⚠️ This **supersedes revision 24's "costs nothing measurable"** and reconciles every apparent disagreement in this document — the metrics were measuring different context regimes, not contradicting each other. ⚠️ Also records that a resume test overwrote R1's top-level model on this date; `checkpoint-4580` is the surviving R1 and every script must point at it.*

*Revision 27 (2026-09-29) — ⚠️ **The two diagnostic jobs revision 26 proposed are withdrawn before running.** Arcee's hybrid scores **55.1 on the same letter-scored lm-eval MMLU (79.0 % relative)**, so the `"A"/"B"/"C"/"D"` indirection is demonstrably retainable through a KDA conversion — their student kept it, ours did not. **No new hypothesis is needed:** 23 % of their tokens, no long-context stage with MLPs unfrozen, and a narrower corpus already account for MMLU, and that is revision 19's conclusion unchanged. The 5-shot figures are recorded as a within-student result and **not quoted as retention**. ⭐ The live hypothesis remains the corpus, and the job that tests it is a **training run on a broadened mix**, not an evaluation.*

*Revision 26 (2026-09-28) — ⚠️⚠️⚠️ **The 5-shot control falsifies revision 20's headline.** R1 at `num_fewshot: 5` **improves on all five metrics** — ARC-C `acc_norm` **0.5205 → 0.5503**, WinoGrande 0.6630 → 0.6780 — where the pre-registered failure condition was "collapses toward chance". **So the student CAN use few-shot exemplars, and a templated 5-shot prompt is not too far off-distribution.** "The dividing line is the shot count" is **withdrawn**: MMLU and GSM8K being the only two 5-shot tasks *and* the only two failures was a coincidence of a five-benchmark sample. ⭐ Replacement hypothesis, flagged as untested: **MMLU is the only benchmark scored on a single letter token** rather than full answer text, so it needs a symbolic indirection the others do not — and GSM8K probably has an unrelated cause. ⚠️ The teacher has never run at 5-shot, so these figures have no relative score yet.*

*Revision 25 (2026-09-28) — ⭐⭐⭐ **The loss components, extracted uniformly for all four runs, and KL is where the mechanisms separate.** Across the four architectures CE spans **1.8 %** and **KL spans 20.6 %** — and **GDN's KL is ~13 % higher than KDA's in both layouts** (+12.8 % hybrid, +13.5 % all-linear), to within 0.7 points. That is the cleanest statement this 2×2 produces, it is behavioural rather than representational, and it is **not** an `hs_cosine` artefact. ⚠️⚠️ **Two corrections.** Revision 24's account of the loss/benchmark disagreement is **RETRACTED**: `hs_cosine` contributes **−12 %** of KDA's layout gap, so it favours the all-linear student; the gap is driven by CE and KL, and the real reason benchmarks miss it is that a **+6.3 % KL** effect is below their resolution. And **R1's run-log start column had been copied from R0** (`7.92 / 10.13 / 0.841`); its own first microbatch is `7.986 / 9.286 / 0.844`, and its **endpoint KL is 0.314, not 0.35**. All four rows now use one extraction method.*

*Revision 24 (2026-09-28) — ⭐⭐⭐ **The 2×2 benchmarked, and one conclusion reverses.** Mean relative score: **`kda_only` 83.64 · `kda_hybrid` 82.60 · `gdn_hybrid` 77.82 · `gdn_only` 76.17**. **KDA beats GDN in all six comparisons** (three benchmarks × two layouts, +4.78 and +7.47 points, pooled Z ≈ 3.1 on sampling error alone) — so the mechanism result survives an independent metric. ⚠️⚠️ **But the LAYOUT conclusion reverses: indistinguishable from zero on benchmarks**, where held-out loss called it clearly worse. `kda_only` matches `kda_hybrid` on quality while carrying a **48.9×** smaller inference state instead of 3.77× — and independently reproduces Arcee's published KDA-only-beats-hybrid finding. ⭐⭐ The disagreement is the finding: the composite objective is **60 % `hs_cosine`**, representational rather than behavioural. **Rank layouts on benchmarks, not on `eval_loss`.** ⬜ Held-out **KL** across the four students remains unmeasured and is the gap that matters.*

*Revision 23 (2026-09-28) — ⭐⭐⭐ **The 2×2 is complete and the mechanism result replicates.** R3 (KDA all-linear, **0.5744**) and R4 (GDN all-linear, **0.5937**) join R1 (**0.5686**) and R2 (**0.5814**). **KDA is ahead in both layouts at all ten eval points — forty numbers in one strict order: `kda_hybrid < kda_only < gdn_hybrid < gdn_only`.** Three findings: the gap **scales with the number of converted layers** (4.74e-4 vs 5.36e-4 per linear layer, so it behaves like a per-layer property); **mechanism outweighs layout 1.77×**, with `kda_only` beating `gdn_hybrid` everywhere; and **all-linear trains SLOWER than the hybrid for both mechanisms**, the model-level confirmation that KDA's advantage is an inference property. All four runs clean, including the all-linear backward pass that had never been exercised. ⚠️ The composite loss is 60 % `hs_cosine` — **held-out CE on all four checkpoints is the missing number.***

*Revision 22 (2026-09-28) — ⭐⭐ **R2 finished and the mechanism comparison has its first number.** Every pre-registered check passed. **KDA ends at held-out combined loss 0.5686 against GDN's 0.5814 — ahead at all ten eval points, with the gap narrowing monotonically from 0.0266 to 0.0128** — and it wins from behind, on 143 M fewer parameters. Since C1 put the two students **0.0022 nats apart at initialisation**, the starting line is not the explanation: this is the first evidence on KDA's own per-channel-gating claim, and it favours KDA. ⭐ **But GDN trained 14 % faster (29,082 vs 25,455 tok/s) on 3.4 % less GPU memory**, which the Track C doc predicted in advance from `_use_fused_qkv_conv`. **The honest framing is a trade, not a win.** ⚠️ n = 1 per cell, one seed.*

*Revision 21 (2026-09-25) — ⭐⭐ **R2, R3 and R4 launched**, completing a **mechanism × layout 2×2** with R1: GDN hybrid (58650590), KDA all-linear (58650592), GDN all-linear (58650595), the last two chained `afterok` behind the first. Every setting but the student and the output path is identical to R1's YAML. Pre-registered pass conditions and the parameter/trainable-tensor counts for each are in "The weekend 2×2" below. ⚠️ None of the three can move open decision 2 — they share R1's corpus and frozen MLPs, so all three should read ~14 % MMLU and ~3 % GSM8K.*

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

⚠️ **This is the archive (rev 42; entries continue from rev 43, latest rev 44).** Live status, open TODOs and the run registry: **`claude/STATUS.md`**.

| Part | What's in it | When you need it |
|---|---|---|
| **1. Stage-1 findings (R1)** | what R1 produced, what it did NOT produce, the GSM8K story, the Arcee comparison, the (decided) open decisions | writing the results chapter |
| **2. The runs** | run log, results, the 2×2, R5 | after each job; writing results |
| **3. Reference** | config settings and why, pre-flight, budget arithmetic | when writing a config; methods chapter |
| **4. Closed** | document isolation — the whole story, settled | methods chapter |
| **5. Traps** | the accumulated list | before every run; limitations |

⚠️ **The sections are in the order they were WRITTEN, not in logical order, and newer sections were often inserted ABOVE older ones.** In the GSM8K part, read "RESOLVED — the GSM8K collapse is the ARITHMETIC CIRCUIT" first; the subsections after it ("The error analysis finally ran … DEGENERATION", "What is still needed", "Is the GSM8K explanation adequate? No") are OLDER (rev 29–31) and superseded. The revision log at the top is the reliable timeline.

### ✅ Index of items that were open here and are now resolved (rev 42)

| was open in section | item | resolved | outcome / where |
|---|---|---|---|
| Status table | GDN sibling student | rev 22–23 | built (Track C) and trained: R2, R4 |
| Status table / Results | IFEval | rev 42 | ✂ **cut for Part II** — base teacher, no chat template |
| What stage 1 did NOT recover | "full 1,319-item GSM8K runs are queued" | rev 20 | R1 0.0243 strict, teacher 0.5353 |
| What stage 1 did NOT recover | cause = frozen MLPs? | open → **tested by stage 2** (rev 42) | — |
| What is still needed (GSM8K) | re-run with `--log_samples` | rev 32 | done on **100** items (not 200), teacher + R1–R4 → loop/termination table in "RESOLVED — … ARITHMETIC CIRCUIT" |
| What is still needed (GSM8K) | same classification for R2–R4 | rev 32 | same table; pre-registration half confirmed (termination exact), half falsified (loop rate) |
| What is still needed (GSM8K) | the teacher's generations | rev 32 | teacher loops 1 %, terminates 98 % |
| What is still needed (GSM8K) | single-step arithmetic probe | rev 32, v2 rev 35 | v1: teacher 76.0 / R1 20.0 / R2 7.5 / R3 24.0 / R4 9.5 %; v2 (2,286 problems, copy control) supersedes it |
| Is the GSM8K explanation adequate? No | "two cheap experiments" | rev 32 | probe done; error tally done as loop/termination classification + probe-v2 error structure |
| MMLU reframes everything | ARC-C at 5-shot | rev 26 | ran: student improves → "shot count" headline withdrawn |
| The 5-shot control ran | teacher never run at 5-shot | rev 27 | decided not needed; 5-shot numbers are within-student only |
| Loss components / 2×2 on benchmarks | held-out KL never measured ("the hole", "still owed") | rev 29 | measured: `test_r5_kl.py` table in "Held-out CE, KL and `hs_cosine`, measured directly" |
| The 2×2 is complete — caveat 1 | held-out CE on all four | rev 28–29 | measured (same table) |
| The 2×2 is complete — caveat 3; R2 section | n = 1 per cell, more seeds | rev 42 | seeds 43 + 44 for R1 and R2 launched 2026-10-08 |
| Open decisions — for Anamaria | 1 budget · 2 scope · 3 ablations | rev 34–42 | 1: stage 1 stays 300 M (R5) · 2: stage 2 = full-parameter KL on all four; corpus broadening cut · 3: GDN 2×2 done; FLP, de-novo cut |
| R5 diverged | `ce_outliers.py` | rev 37 | latent, input-triggered damage (massive activation from layer 3; layer-0 projections 4–6× R1) |
| R5 diverged | pre-registration B "not interpretable" | rev 37 | explained by the outlier sequences |
| R5 diverged | batching check, `SET=b1` | rev 36 | ~4.5 % of answers move; R5 − R1 +7.6 → +7.4, conclusions stand |
| What this does to stage 2 | "peak LR 3e-4 or lower" | rev 39–42 | superseded: stage 2 constant 1e-5, plus cosine 5e-5 → 1e-5 check on R1 |
| Reference | `save_total_limit: 1` is the wrong setting | rev 33–38 | applied: R5 and stage 2 keep 10 checkpoints |
| Rev 41 | `stage2_prep.sbatch`, positions to 1,849,999 | rev 42 | split v4 (duplicates skipped, to 1,864,858), warm-up by hand |
| Traps | "the single-step probe has never been run" | rev 32 | it ran |
| STATUS ⬜? | R5 MMLU / GSM8K (`eval_mml` / `eval_gsm` jobs) | rev 43 | MMLU 0.3068 (= R1), GSM8K strict 0.0409 — table under R5's 0-shot section |

**Still open** (tracked in `STATUS.md`, not here): B9b inference speed · retrieval probe · HumanEval/MBPP · evaluation of stage 2, seed replicates and the LR check · whether the 2× off-distribution gap is architectural (needs a non-converted control; likely stays a limitation).

---

# PART 1 — STAGE-1 FINDINGS (was "NOW" until rev 41)

## Status *(frozen at rev 42 — live status is in `claude/STATUS.md`)*

| | |
|---|---|
| Students | ✅ four, rebuilt from `d7cd0a0`, build verified deterministic |
| Pipeline | ✅ R0 clean, **R1 clean** |
| Document isolation | 🔵 closed — path (a), see Part 4 |
| **R1 — stage 1, 300 M tokens** | ✅ **COMPLETE, job 58141472** |
| Teacher baseline | ✅ **all five** — ARC-C / HellaSwag / WinoGrande / MMLU / GSM8K |
| **Student benchmarks** | ✅ **all five** + held-out CE — and **MMLU is the headline**, see below |
| **Generation** | ✅ **Track B closed** — `claude/track_b.md` |
| **GSM8K / IFEval** | ✅ GSM8K full (1,319) · ✂ IFEval cut for Part II (rev 42) |
| B9b decode throughput | ⬜ the speed half of the architectural claim — **still open, see STATUS.md** |
| GDN sibling student | ✅ built and trained — R2 (hybrid), R4 (all-linear) |
| R2–R4 (2×2) | ✅ complete, benchmarked, held-out CE/KL measured |
| R5 (1 epoch) | ⚠️ finished but unstable — latent damage (rev 37); not used as a base |
| Stage 2, R1–R4 | 🔶 launched 2026-10-08 (rev 42) |
| Seed replicates R1/R2 × {43, 44} | 🔶 launched 2026-10-08 (rev 42) |
| R1 stage-2 LR check (cosine 5e-5) | 🔶 queued 2026-10-12 |

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

⚠️ **`--limit` is a pipeline test, not a measurement.** 0/20 is consistent with a true rate up to ~14 %. The full 1,319-item runs are queued; **do not quote 0.00 until they land.** ✅ *(RESOLVED rev 20: full run 0.0243 strict.)*

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
| The cause is the frozen MLPs specifically | ⬜ **not established, and now the weaker of the two** — *(rev 42: tested by stage 2, which unfreezes them)* |
| The cause is the corpus (FineWeb-Edu is narrow) | ⭐ **best supported** — it is what a distribution-tracking gap predicts |
| The 2× is attributable to the KDA architecture rather than to ordinary generalisation | ⬜ **not established** — would need the same recipe on a non-converted student |

**The defensible sentence is:** *the student reproduces its teacher about twice as poorly off its training distribution as on it, on two unrelated out-of-distribution corpora; GSM8K sits off that distribution and is multiplicatively sensitive to per-token error.*

⭐ **That is still enough to carry open decision 2 — but it points at a different lever than revision 18 thought.**

⚠️ **And language modelling on mathematical text is not the ability to solve a problem.** A model can predict the tokens of a worked solution it could not have produced. This localises representational damage; it is not a second GSM8K.

## ⭐⭐⭐ RESOLVED — the GSM8K collapse is the ARITHMETIC CIRCUIT, and it is now measured

*Single-step arithmetic probe (n=200 per model, 4-shot, `"What is a OP b? Answer:"`) and GSM8K loop/termination rates (n=100 documents per model, `--log_samples`). 2026-09-29.*

### The measurement that settles it

| | **single-step arithmetic** | GSM8K strict (full 1,319) |
|---|---|---|
| **teacher** | **76.0 %** | 53.53 % |
| **R1 kda_hybrid** | **20.0 %** | 2.43 % |
| R2 gdn_hybrid | 7.5 % | 1.36 % |
| R3 kda_only | 24.0 % | 0.83 % |
| R4 gdn_only | 9.5 % | 0.08 % |

⭐⭐⭐ **And it predicts GSM8K with a single parameter fitted on the teacher alone.** Treating a problem as *k* independent operations:

```
teacher:  0.76^k = 0.5353   ->   k = 2.28 effective operations per problem
then, with no further fitting:
  R1 kda_hybrid:  0.20^2.28 = 2.56 %       measured strict = 2.43 %
```

**2.56 % predicted against 2.43 % measured.** ⚠️ It fits R1 and R4 and not R2 (0.27 % vs 1.36 %) or R3 (3.88 % vs 0.83 %) — the probe carries ±2.9 % on *p*, which `p^2.28` amplifies sharply. **The defensible claim is narrow and strong: the main student's GSM8K score is fully accounted for by its single-step arithmetic.**

### ⭐⭐ This is a PRECISION failure, not a retrieval failure

The probe prompt is four exemplars plus *"What is 437 + 62? Answer:"* — **no long context at all.** Adding two three-digit numbers needs exact digit-position alignment and carry propagation: attention can address "the tens digit of the first operand" precisely, a fixed-size recurrent state carries a lossy summary of everything at once. **The damage is to precision, and it is visible with a 40-token prompt.**

### ⭐⭐⭐ So there are TWO independent damages, on two different axes

| | tracks | size |
|---|---|---|
| **arithmetic / precision** | **MECHANISM** — KDA 20–24 %, GDN 7.5–9.5 % | **~2.5×**, the largest mechanism effect anywhere in this project — bigger than the 22 % on held-out KL |
| **format / termination** | **LAYOUT** — 56 % → 45 % → 31 % → **1 %** | teacher 98 % → R4 1 % |

Layout barely touches arithmetic (20 vs 24, 7.5 vs 9.5 — all-linear marginally *better*). Mechanism barely touches termination. **They are separate effects and belong in separate sentences.**

### The loop / termination table

| | loop | **terminates `####`** | strict | flexible |
|---|---|---|---|---|
| teacher | **1.0 %** | **98.0 %** | 49 % | 49 % |
| R1 kda_hybrid | 39.0 % | **56.0 %** | 1 % | 2 % |
| R2 gdn_hybrid | 51.0 % | 45.0 % | 0 % | 3 % |
| R3 kda_only | 32.0 % | 31.0 % | 1 % | 2 % |
| R4 gdn_only | 42.0 % | **1.0 %** | 0 % | 8 % |

✅ **Pre-registration, half confirmed and half falsified.** *"The `flexible/strict` ratio is a format-compliance meter"* — **confirmed exactly**: termination 56 > 45 > 31 > 1 is the precise inverse of 1.44× < 1.73× < 3.93× < 40.75×. *"Loop rate rises with the all-linear layout"* — **falsified**: R3 loops **less** than R1 (32 % vs 39 %), order R3 < R1 < R4 < R2. ⭐ **The mechanism was right and the proxy was wrong** — R4 mostly does not repeat, it rambles without ever emitting `####`. **Termination is the real quantity; looping is one of several ways to miss it.**

### ⚠️⚠️ RETRACTED: revision 30's "degeneration, not arithmetic", and revision 31's multiplicative decomposition

Revision 31 split the 32.7× gap into "1.75× from not finishing × 18.7× from finishing and being wrong". **That is arithmetically valid and causally meaningless**, for three reasons:

- **Looping is not a separate cause — it is what failure looks like.** A model that cannot compute has nothing to write, so it repeats. Multiplying the two implies an independence that does not exist.
- **Conditioning on "terminates" conditions on a post-treatment variable.** The generations that reach `####` are the easier problems and the luckier runs, so the conditional accuracy is measured on a favourably selected subsample.
- ⭐ **R4 refutes it empirically: 1 % termination, 8 % flexible.** If termination gated correctness, R4 would score zero.

⭐⭐ **The correct account is single-cause.** The arithmetic circuit is damaged (76 % → 20 %); a model that cannot compute produces nothing worth terminating, so it rambles or loops; the few that do terminate are mostly still wrong. **Degeneration is a symptom, not an additional cause** — and revision 30 mistook it for the disease because ten generations showed the symptom vividly and the cause not at all.

*Trap: **a decomposition can be exactly right and mean nothing.** `P(A) = P(B)·P(A|B)` always holds; it describes a mechanism only when B is not itself caused by the thing you are explaining.*

---

### ⭐⭐⭐ The error analysis finally ran — and **the dominant failure is not arithmetic, it is DEGENERATION**

*Ten consecutive generations (`doc_id` 10–19) from R1's GSM8K sample log, read individually. 2026-09-29.*

| | count |
|---|---|
| **collapse into verbatim repetition, never emit `####`** | **6 / 10** |
| terminate with `####` but answer wrong | 4 / 10 |

⭐⭐⭐ **Six of ten generations do not fail at mathematics. They fail to stop.** Doc 15 repeats *"The merchant would make a profit of $5,000 if he chooses jewelry."* **sixteen times verbatim** and never attempts a calculation. Doc 14 repeats *"80 % of 80 % = 0,8"* six times. Doc 11 emits `55 + 55 + 55 + ...` to the token cap. **This is not visible in a score; it is only visible in the text.**

#### The arithmetic is mediocre, not broken

Every verifiable operation across the ten:

| ✅ correct | ❌ wrong |
|---|---|
| `2×80=160` · `6×55=330` · `180/4=45` · `3×7=21` · `20×35=700` · `150+150=300` · `220+300=460` · `460+300=660` · `1.5×7=10.5` | `3×68=216` *(204)* · `80+150=220` *(230)* · `1.5×7=9.5` *(10.5)* · `10.5×7=77.5` *(73.5)* · `60+180+45=225` *(285)* · `700+1050=1350` *(1750)* |

**Nine right, six wrong — about 60 % per operation.** ⭐ Doc 12 computes `1.5 × 7` **wrong and then right in consecutive lines**.

### ⭐⭐ And this is the first account that reproduces the observed number

| model | prediction |
|---|---|
| arithmetic alone, 4-step chain | requires per-step **0.395** — far below the **0.60** observed |
| **arithmetic × termination**: `0.60⁴ × 0.40` | **0.0518** |
| **observed R1 strict** | **0.0243** |

**Same order of magnitude, from two independently observed rates.** Neither alone works: 60 % arithmetic on a four-step chain predicts 13 %, and a 40 % termination rate alone predicts 40 %. **Together they land within a factor of two of the measured score.** ⚠️ *n = 10, the four-step assumption is rough, and 0.0518 against 0.0243 is still a factor of 2. This is a decomposition that works, not a fit.*

### ⭐⭐⭐ Why this unifies everything in this document

**Repetition collapse is a long-range retrieval failure.** Not repeating yourself requires knowing precisely what you have just emitted — attention can look at it exactly; a fixed-size recurrent state carries a lossy summary. So:

| observation | same cause |
|---|---|
| 6/10 generations loop | can't track its own recent output |
| **`flexible/strict` ratio 1.44× → 1.73× → 3.93× → 40.75×** | the all-linear students loop **far** more — `gdn_only` emits `####` once in 1,319 |
| **MMLU hybrid > all-linear at 4.3–4.7 σ** | 5-shot prompts need retrieval across five exemplars |
| no layout effect on 0-shot benchmarks or held-out CE/KL | short contexts, nothing to retrieve |

⭐⭐ **The `flexible/strict` ratio is a repetition meter.** It was already in the results table; nobody knew what it was measuring until the generations were read.

#### Three more failures, each a retrieval failure of a different kind

- ⭐ **Doc 18 hallucinates "$15 per dozen of eggs"** — there is no $15 in the question. **Exemplar 5 of the same prompt says "$15 per hour".** It pulled a number from the wrong place in a 2,000-token context.
- ⭐ **Docs 10 and 13 write one operation in prose and a different one inside `<<>>`**: *"180 × 3/4 = `<<180/4=45>>`"*, *"2/3 × 5 = `<<2*5=10>>`"*. The annotation format requires copying operands from the sentence just written. **A short-range copy, failed.**
- ⭐ **Doc 13 answers "Melanie started with 5 vacuum cleaners"** — restating the *given* (5 left) as the *answer*.

📌 **Doc 19 scores `exact_match 1.0` and is not a success** — it loops twelve times on *"so the average speed is 6 miles per hour"*, and `flexible-extract` takes the trailing 6, which happens to be right. **A correct GSM8K score does not imply a correct generation.**

### ✅ RESOLVED (rev 32) — ~~What is still needed~~

> ✅ **All four rows were done in revision 32** — results in "RESOLVED — the GSM8K collapse is the ARITHMETIC CIRCUIT" above: the `--log_samples` re-run on **100** items per model (not 200) for the teacher and R1–R4, classified for loops and termination; and the single-step arithmetic probe, n = 200 per model (replaced by probe v2 in revision 35). The table is kept as the plan that was made.

| | why |
|---|---|
| **200 items per model, re-run with `--log_samples`** | ⚠️ **the full runs wrote no samples — see the trap below — and the generations are unrecoverable.** 200 is enough: a loop rate is a **proportion** (±3.5 % at p≈0.6), not a rare event, so this is a legitimate use of `--limit` where quoting `exact_match` from it would not be. `--limit` takes the first N deterministically, so all five models see the same problems — a paired comparison |
| **The same classification for R2, R3, R4** | pre-registered prediction: **loop rate rises monotonically hybrid → all-linear**, matching the `flexible/strict` ratio |
| **The teacher's generations** | the teacher loops too (B7c caught it repeating *"Paris is the capital of France"* six times). **Without the teacher's loop rate, "60 %" has no baseline.** |
| Single-step arithmetic probe | still worth it — the 60 % here is confounded with misreading the question |

⚠️⚠️ **This supersedes every earlier account in this document.** "The student reproduces the form perfectly and fails the arithmetic" was written from three cherry-picked generations and is **wrong about the majority of cases** — most do not reproduce the form at all, because they never terminate.

### ⚠️⚠️ Is the GSM8K explanation adequate? **No** — stated plainly, 2026-09-29

> ✅ **Superseded in revision 32: yes, now it is.** The single-step arithmetic probe accounts for R1's GSM8K score (`0.20^2.28 = 2.56 %` predicted vs 2.43 % measured). Kept as the reasoning that led there.

This document has offered three things as the account: multiplicative composition along a chain, distance from the training distribution, and budget-plus-frozen-MLPs. **Only the last is quantitatively serious, and none of it has been tested directly.** Where it actually stands:

**1. ⚠️ The multiplicative argument is too weak by an order of magnitude.** A GSM8K solution needs roughly four arithmetic steps. If each succeeds independently with probability *p*:

| | score | implied per-step *p* |
|---|---|---|
| teacher | 0.5353 | **0.855** |
| R1 student | 0.0243 | **0.395** |

**Per-operation accuracy would have to fall from 86 % to 40 %.** That is not what a **17 % perplexity penalty** does — a penalty that size buys a few points on a four-step chain, not a 22× collapse. **"Errors compose multiplicatively" explains why the number is 4 % rather than 40 %; it does not explain why per-step accuracy halved.**

**2. ⭐ The two recorded errors are not arithmetic errors — they are operand copies.** *(Superseded by the ten-generation analysis above, which finds a larger and different dominant failure.)* `150,000 − 120,000 = 150,000` and `2 + 0.5 = 2` both return the **first operand unchanged**. That is a retrieval-and-alignment failure, not a noisy computation. *(And `<<16*7=112>>` in the same sample is correct three-digit multiplication.)* ⚠️ **n = 2 cherry-picked generations. This is a hypothesis-generating observation, not evidence.**

**3. ⚠️ We have never measured whether the student can do ONE arithmetic operation.** ✅ *(Measured in rev 32: teacher 76 %, R1 20 %.)* Every claim about "the arithmetic" rests on three sampled generations from a 20-item smoke run. **The single most obvious experiment in this whole line of inquiry has not been run.**

**4. ⭐⭐ The story that fits, offered as a story:** multi-digit arithmetic is an **attention-heavy, position-precise** circuit — carrying and place-value alignment need exact lookup of specific digit positions. We replaced **27 of 36** token-mixing layers with a mechanism that carries a lossy fixed-size summary instead, left the MLPs frozen so they still expect attention-shaped inputs, and retrained on **FineWeb-Edu, which contains almost no worked arithmetic**. Both halves are needed: the circuit is broken in 27 places *and* the corpus never exercises it. ⭐ It also sits naturally beside the layout finding above — the all-linear students lose the `####` format 40× more often, which is the same class of failure.

✅ *(Both done — probe: rev 32 and v2 rev 35; error tally: loop/termination classification on 100 items, rev 32.)* **Two cheap experiments would replace the story with evidence:**

| | what it settles |
|---|---|
| **Single-step arithmetic probe** — few-shot `"What is 150000 - 120000?"` over a few hundred generated pairs, teacher and all four students | If single-step accuracy is near the teacher's, the failure is composition and the chain argument needs rescuing. **If single-step is already at 40 %, the arithmetic circuit itself is broken** and the chain argument is irrelevant. *A script, not a queue job.* |
| **Error-type tally** — re-run GSM8K with `--log_samples` and classify the 1,319 generations: operand copy, wrong operation, right method / wrong digits, no `####`, non-termination | Turns "it fails the arithmetic" into a distribution. The operand-copy hypothesis predicts one bucket dominates. |

⚠️ **Until one of those runs, the thesis should say what is measured and stop there:** *the student retains 79–89 % of the teacher on three 0-shot benchmarks and 4.5 % on GSM8K; the mechanism of that collapse has not been established.* That is a defensible sentence. "Errors compose multiplicatively" is not, at these magnitudes.

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

⚠️⚠️ **RAN 2026-09-28 — and the answer was neither. The student IMPROVES at 5-shot, so both (a) and (b) are falsified and this whole section's framing is withdrawn. See "The 5-shot control ran" above.**

⭐⭐ ~~One cheap job separates them: ARC-Challenge at 5-shot.~~ If a benchmark the student holds at 79.1 % 0-shot collapses toward chance at 5-shot, the fault is the *format*, not the task — and (a)/(b) become the live question. If it holds, the collapse is task-specific and MMLU's breadth is implicated instead.

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

> ✅ **All three decided (rev 34–42).** 1. Stage 1 stays at 300 M tokens — R5's 7.6× tokens gave no 0-shot gain and diverged. 2. Stage 2 = RADLADS step 2 (full-parameter KL, constant 1e-5, 1 B tokens) on all four students, launched 2026-10-08; corpus broadening cut → future work. 3. GDN 2×2 done (R2–R4); FLP and de-novo cut. The original text is kept below.

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
| **R1** ⭐ | 09-18/19 | **58141472** | 4 | ⚠️ *repo* 3fb1104 / 2e26723, **model 3aafffc** | **300.1 M** | **4580** | **7.986 / 9.286 / 0.844 → 2.261 / 0.314 / 0.069** | 10.73 | ✅ **3.27 h, 25,455 tok/s, clean** |
| **R2** ⭐ | 09-25/26 | **58650590** | 4 | **8f40624**, clean | **300.1 M** | **4580** | **7.981 / 9.349 / 0.840 → 2.283 / 0.355 / 0.070** | 10.36 | ✅ **2.87 h, 29,082 tok/s, clean.** GDN hybrid 3:1; held-out **0.5814** |
| **R3** | 09-26/27 | **58650592** | 4 | **8f40624**, clean | **300.3 M** | **4580** | **8.590 / 10.040 / 0.837 → 2.276 / 0.334 / 0.068** | 10.11 | ✅ **3.61 h, 23,081 tok/s, clean.** KDA all-linear; held-out **0.5744** |
| **R4** | 09-26/27 | **58650595** | 4 | **8f40624**, clean | **300.1 M** | **4580** | **8.531 / 9.914 / 0.846 → 2.303 / 0.379 / 0.075** | 10.60 | ✅ **3.04 h, 27,402 tok/s, clean.** GDN all-linear; held-out **0.5937** |
| R5-a | 10-01 | 59074838 | 4 | — | 0 | 0 | — | — | ⚠️⚠️ **crash**: NCCL barrier timeout at 600 s while rank 0 tokenized 2.921 M docs (606 s projected). See below |
| **R5** | 10-02/04 | **59234295** | 4 | 2e26723 / model 7291d97 (`dirty: false`) | **2.2899 B** | **34940** | **8.093 / 9.904 / 0.845 → 2.225 / 0.329 / 0.073** | 9.99 | ⚠️⚠️ **finished but UNSTABLE**: eval_loss spiked to 1.742 and 1.755, `grad_norm` max **83,070**; recovered to **0.5715** (R1 0.5686). 23.98 h, **26,522 tok/s**, tokens agree two ways, 10/10 checkpoints. See below |
| S2 smoke | 10-08 | 59735584 | 4 | R1 ckpt-4580 | — | 20 | KL 0.18–0.57 at start (R1's level) | 13.8 | ✅ guards, no `Froze` line, weights-only save works, ~23 k tok/s |
| **R1-S2** | 10-08/09 | **59749921** | 4 | R1 ckpt-4580 | **1.0005 B** | **15260** | KL (train `loss`) 0.344 → **0.276** | 12.82 | ✅ **11.83 h, 23,491 tok/s, clean.** Held-out KL (`eval_loss`, KL only) **0.3379 → 0.2731**, monotone; max `grad_norm` 15.95 @ ep 0.944; no `Froze` line (correct) |
| **R2-S2** | 10-08 → ✅ | **59749924** | 4 | R2 ckpt-4580 | 1.0 B | 15260 | KL → see notes | — | ✅ **11.42 h, 24,330 tok/s, clean.** Held-out KL **0.3726 → 0.2965**; max `grad_norm` 18.75 |
| **R3-S2** | 10-08 → ✅ | **59749930** | 4 | R3 ckpt-4580 | 1.0 B | 15260 | KL → see notes | — | ✅ **12.82 h, 21,668 tok/s, clean.** Held-out KL **0.3563 → 0.2937**; max `grad_norm` 8.46 |
| **R4-S2** | 10-08 → ✅ | **59749932** | 4 | R4 ckpt-4580 | 1.0 B | 15260 | KL → see notes | — | ✅ **11.32 h, 24,535 tok/s, clean.** Held-out KL **0.4022 → 0.3298**; max `grad_norm` 9.13 |
| R1-s43 / R1-s44 | 10-08 → | see STATUS.md | 4 | init student | 300.3 / 300.1 M | 4580 | 7.767 / 9.760 / 0.833 → 2.321 / 0.283 / 0.068 · 8.161 / 8.759 / 0.843 → 2.294 / 0.310 / 0.069 | 9.99 | ✅ **3.26 / 3.27 h, 25,600 / 25,518 tok/s, clean.** Held-out **0.5697 / 0.5693** (R1: 0.5686) |
| R2-s43 / R2-s44 | 10-08 → | see STATUS.md | 4 | init student | 300.2 / 300.2 M | 4580 | 7.793 / 9.885 / 0.835 → 2.339 / 0.315 / 0.067 · 8.155 / 8.821 / 0.840 → 2.317 / 0.349 / 0.071 | 10.36 | ✅ **2.79 / 2.81 h, 29,901 / 29,696 tok/s, clean.** Held-out **0.5814 / 0.5813** (R2: 0.5814) |
| R1-S2-cos | 10-12 → | see STATUS.md | 4 | R1 ckpt-4580 | 1.0 B | 15260 | — | — | ✅ **11.86 h, 23,429 tok/s, clean — but worse:** held-out KL **0.5282 → 0.2956** (constant 1e-5: 0.2731); max `grad_norm` 11.48 |

⚠️ *Stage-2 `eval_loss` is **KL only (T = 2)**; stage-1 `eval_loss` is the composite 0.2 CE + 0.2 KL + 0.6 hs_cos. Both are on the same 200 held-out documents, but **they are different quantities** — compare stage-2 runs only with each other (rev 43).*

⚠️ *A duplicate R2 (58650570) was submitted and cancelled before it started. **Two jobs sharing one `output_path` is silent corruption**, not an error — `squeue -u $USER` after every submission burst.*

*torch 2.12.0+cu126, transformers 5.8.1, fla 0.5.2, deepspeed 0.19.6 throughout.*

✅ **All four rows now use ONE method: first logged microbatch → mean of the last twenty**, extracted from each run's own `.out` with the same script.

⚠️⚠️ **R1's start column was wrong until 2026-09-28, and the error is instructive.** It read `7.92 / 10.13 / 0.841` — **byte-identical to R0's row**, because it had been copied from there. Two runs on different data samples cannot agree to three significant figures. R1's own first microbatch is **7.986 / 9.286 / 0.8441**. *Trap: **when two rows of a table match exactly, suspect a copy before you celebrate a reproduction.***

⚠️ **R1's end column changed too, from `2.26 / 0.35 / 0.070` to `2.261 / 0.314 / 0.069`.** The old KL came from the **epoch-0.7 bin** of the curve below, and the run continued to epoch 0.7655 with KL still falling. **Both numbers are correct measurements of different points; only the endpoint belongs in this table.** ⚠️ Consequence: **R1's endpoint KL is 0.314, not 0.35** — the Arcee comparison (≈ 0.2) should quote 0.314.


⚠️ **The two hashes in R1's row are not the same thing, and this is a trap.** The sbatch printed **3fb1104** from `git log -1`; the model that trained is the *checkpoint's* frozen copy, **3aafffc**. Proof from the run's own log: 48 `[AFMKDA-PROBE]` lines appear, and those prints had already been deleted from the repo. **The provenance line records the repo, not the code.**

⭐ **Fixed since.** The provenance block now prints `build_info.json` and `sha1sum $STUDENT/*.py`, and `afm_kda` is committed clean at **`d7cd0a0`** with `logs/` and `__pycache__/` gitignored so `git status --porcelain` is a meaningful signal.

## Results

| Metric | Teacher | At init | **After stage 1 (R1)** | After stage 2 *(R1-S2 finished; evaluation queued — see STATUS.md / results.md)* |
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
| IFEval | ✂ cut for Part II (rev 42) | — | — | |
| **CE gap on MATH text** | — | — | **+0.3372** (2.11× the web gap) | |
| Parameters | 4.6192 B | 4.9990 B | 4.9990 B | |
| Throughput @ 4 nodes | — | — | **25,455 tok/s** | |

**Teacher baseline, job 58216691**, 6 minutes on the debug queue. Protocol frozen in `claude/eval_zeroshot_afm.sbatch`, `claude/eval_mmlu_afm.sbatch` and `claude/eval_generative_afm.sbatch`: 0-shot for ARC-C, HellaSwag and WinoGrande; 5-shot for MMLU and GSM8K; 0-shot IFEval; no chat template. ✅ **ARC-C, HellaSwag and GSM8K all land on the blog post's published teacher figures**, which validates the harness configuration three independent ways.

## ⭐⭐ The weekend 2×2 — launched 2026-09-25

Three runs queued, chained `afterok` behind R2 so a shared failure costs one run instead of three. With R1 already in place they complete a full cross of **mechanism × layout**:

| | **KDA** | **GDN** |
|---|---|---|
| **hybrid 3:1** (27 linear + 9 attention) | **R1** ✅ done | **R2** ✅ **done** |
| **all-linear** (36 linear) | **R3** ✅ **done** | **R4** ✅ **done** |

⭐ **Every setting except the student and the output path is byte-identical to R1's YAML** — verified by a three-line `diff` on each. Same corpus, same 4 nodes, same `grad_accum: 2`, same `max_steps: 4580`, same schedule. **That is what makes the cells comparable**; any other difference would contaminate the only comparison the thesis is built on.

Cost: ~10 h wall-clock, ~1,260 local hours — about 1 % of the remaining budget.

### Pre-registered pass conditions

Written down before the runs, so checking is matching rather than judging:

| | R2 (gdn hybrid) | R3 (kda only) | R4 (gdn only) |
|---|---|---|---|
| `PREWARM params` | 5,141,828,536 | 5,046,996,688 | 5,237,399,968 |
| `Froze N tensors` | **147**, ×16 | **147**, ×16 | **147**, ×16 |
| trainable tensors | 9×5 + 27×13 = **396** | 36×16 = **576** | 36×13 = **468** |
| step-0 `cross_entropy` | ≈ **7.92** | ≈ 7.9 | ≈ 7.9 |
| `roll back to CPU` | **0** | 0 | 0 |

⭐ **147 is the same in all three because the frozen set — embeddings, `lm_head`, the final norm, 72 layernorms and 72 MLP tensors — contains nothing mechanism-specific or layout-specific.** If any run prints something else, `frozen_res` did not transfer and that run's numbers are not comparable to R1's. Kill it.

⚠️ **One thing genuinely untested: neither `only` student has ever had a backward pass.** Forward is covered (B6e and B7 both ran `afm_kda_only_init`) and the construction is shared, so the risk is low — and a failure would land within two minutes, costing about four local hours.

⚠️ **R2, R3 and R4 cannot settle open decision 2.** All three share R1's corpus and frozen MLPs, so all three are expected to read ~14 % on MMLU and ~3 % on GSM8K. **This cross measures mechanism and layout, not the missing capability** — which is exactly why those two axes being at the floor matters.

## ⚠️⚠️⚠️ The 5-shot control ran, and it FALSIFIES revision 20's headline

*R1 (`checkpoint-4580`) on the same three tasks at `num_fewshot: 5`, 2026-09-28. This was pre-registered in revision 20 as the job that separates the two surviving explanations. It did something neither of them predicted.*

| R1 student | 0-shot | **5-shot** | Δ |
|---|---|---|---|
| ARC-C `acc` | 0.4778 | **0.5119** | **+0.0341** |
| ARC-C `acc_norm` | 0.5205 | **0.5503** | **+0.0298** |
| HellaSwag `acc` | 0.5449 | **0.5480** | +0.0031 |
| HellaSwag `acc_norm` | 0.7240 | **0.7297** | +0.0057 |
| WinoGrande `acc` | 0.6630 | **0.6780** | **+0.0150** |

⭐⭐⭐ **The student does not collapse at 5-shot. It improves, on all five metrics.** The pre-registered failure condition was *"collapses toward chance"* — chance on ARC-C is 0.25, and it went from 0.52 to **0.55**.

### What this kills

Revision 20 built its headline on the observation that **both** collapsed benchmarks (MMLU 13.9 %, GSM8K 4.5 %) were the **only two 5-shot tasks**, and the three that held were all 0-shot. It offered two mechanisms:

| | status |
|---|---|
| **(a) the student cannot use few-shot exemplars held at distance** | ❌ **FALSIFIED** — five exemplars make it *better*, not worse |
| **(b) a templated 5-shot prompt is too far from raw FineWeb-Edu** | ❌ **FALSIFIED** — same templated format, same five-exemplar block, +3.0 points |

⚠️ **So "the dividing line is the shot count" is withdrawn.** It was the best available reading of five numbers; a sixth killed it. The two collapsed benchmarks being the two 5-shot ones was a **coincidence of a small sample**, and revision 20 over-read it — the same error, in the same document, that the Tulu 3 control caught in revision 19.

### ⚠️ What this does NOT establish

**The teacher has never been run at 5-shot** *(✅ decided rev 27: not needed — these numbers are reported within-student only)*, so there is no relative score for these numbers. If the teacher gains +6 points where the student gains +3, relative retention still falls — just nowhere near to 14 %. **The absolute non-collapse is solid; the retention question is open and costs one job.**

### ⭐ The hypothesis that replaces it, and why it is testable

**MMLU is the only one of the five benchmarks scored on a single letter token.** In lm-eval, `mmlu` presents enumerated options and scores `"A"` / `"B"` / `"C"` / `"D"` as the continuation; ARC-Challenge, HellaSwag and WinoGrande all score the **full text** of each choice. So MMLU additionally requires the symbolic indirection *"the right answer is the third one → emit the token ' C'"* — a narrow, format-specific competence, not a language-modelling one.

📌 **Stated as a hypothesis, not a measurement**, and it does **not** cover GSM8K, which is free generation of digits and already has its own explanation (multi-step arithmetic composing multiplicatively). **We may simply have two unrelated failures that a five-benchmark sample made look like one pattern.**

### ⭐⭐ …and Arcee's own numbers largely close the letter-token hypothesis too

**Their hybrid scores 55.1 on the same lm-eval MMLU, with the same `"A"`/`"B"`/`"C"`/`"D"` scoring — 79.0 % relative.** So the letter indirection is plainly **retainable** through a KDA conversion. It is not an inherent barrier that the architecture imposes; **their student kept it and ours did not.**

⭐⭐ **Which returns the explanation to where revision 19 left it, and where it should have stayed:** ours had **23 % of their tokens**, **no long-context stage with the MLPs unfrozen**, and a **narrower corpus**. MMLU is simply the benchmark most sensitive to all three. **No new hypothesis is needed and no diagnostic job is warranted** — an earlier draft of this section proposed running the teacher at 5-shot and MMLU at 0-shot, and **both would have tested a loop this document opened rather than the explanation it actually holds.**

⚠️ **Consequence for the 5-shot numbers above: record them, do not quote them as retention.** "The student does not collapse at 5-shot" is the finding and needs no teacher run. The frozen protocol is unchanged — ARC-C / HellaSwag / WinoGrande at 0-shot and MMLU at 5-shot remain the numbers of record.

⭐ **The live hypothesis is still the corpus, and the job that tests it is a training run, not an evaluation:** 300 M tokens on a broadened mix, everything else identical to R1. *Trap: **a falsified hypothesis generates tidy follow-up work that tests nothing.** Ask what the next job would change your mind about before queueing it.*

---

## ⭐⭐⭐ Held-out CE, KL and `hs_cosine`, measured directly — and what they settle

*`test_r5_kl.py`, 64 × 2048 = 131,072 held-out tokens, all three components replicating DistillKit's own source (`kl.py`, `hidden_state.py`). 2026-09-29.*

| | CE | **KL** | `hs_cosine` |
|---|---|---|---|
| **teacher (self)** | **2.2963** | **0.0000** | **−0.0000** |
| **R3 kda_only** | **2.4379** | **0.3328** | **0.0711** |
| R1 kda_hybrid | 2.4559 | 0.3568 | 0.0754 |
| R4 gdn_only | 2.4623 | 0.3779 | 0.0785 |
| R2 gdn_hybrid | 2.5082 | 0.4357 | 0.0835 |

⭐⭐ **The teacher row validates three quantities at once — CE 2.3030 on the first 8 (A5b's published figure), KL exactly 0, `hs_cosine` exactly 0.** Three independent zeros from one identity check. The arithmetic matches DistillKit's.

### ✅ Mechanism: robust, and KL is the discriminator

| GDN − KDA, relative | CE | **KL** | `hs_cosine` |
|---|---|---|---|
| hybrid | +2.1 % | **+22.1 %** | +10.7 % |
| all-linear | +1.0 % | **+13.6 %** | +10.4 % |

**All three components favour KDA, in both layouts.** ⭐ **KL separates by an order of magnitude more than CE does** (22 % against 2 %), which is the practical lesson: *cross-entropy barely distinguishes four architectures that differ visibly in how well they reproduce the teacher's distribution.* Report KL.

### ⚠️⚠️ And the layout effect on held-out loss does NOT survive a change of sample

Recomposing `0.2·CE + 0.2·KL + 0.6·hs_cosine` from the table above and setting it beside DistillKit's own `eval_loss` — a **different** draw from the same held-out split (200 shuffled documents, EOS-inserted, TRL-packed):

| | composite, this sample | DistillKit `eval_loss` |
|---|---|---|
| R1 kda_hybrid | 0.6078 | **0.5686** |
| R3 kda_only | **0.5968** | 0.5744 |
| R2 gdn_hybrid | 0.6389 | **0.5814** |
| R4 gdn_only | **0.6151** | 0.5937 |

| layout effect (all-linear − hybrid) | this sample | DistillKit |
|---|---|---|
| KDA | **−0.0110** | **+0.0058** |
| GDN | **−0.0237** | **+0.0123** |

⭐⭐⭐ **The sign flips, for both mechanisms, between two samples of the same held-out split.** **So the layout effect on short-context language modelling is at or below between-sample noise** — not "all-linear is slightly better", as this document said one revision ago, and not "slightly worse" either. **It is not reliably measurable at this effect size.**

⭐ **That sharpens the context-length finding rather than weakening it.** Short-context language modelling: no reliable layout difference. MMLU 5-shot: **4.7 σ and 4.3 σ** in favour of the hybrid. GSM8K format compliance: **1.44× → 40.75×**. The hybrid's nine attention layers buy nothing you can measure on packed web text and a great deal on a long prompt.

📌 **The mechanism axis is untouched by this**, because it is the same sign and a far larger magnitude in both samples and on every benchmark.

⚠️ *Trap recorded: **two draws from the same held-out split are not the same measurement.** A 0.01 effect on a quantity near 0.6 needs either a paired design or an effect an order of magnitude larger before a sign can be trusted.*

---

## ⭐⭐⭐ The complete 2×2 — and the layout effect turns out to depend on CONTEXT LENGTH

*All five benchmarks, held-out CE and held-out KL, for all four students. 2026-09-29.*

| relative score | **MMLU** 5-shot | **GSM8K** 5-shot | ARC-C 0-shot | HellaSwag 0-shot | WinoGrande 0-shot | *0-shot mean* | held-out CE | held-out **KL** |
|---|---|---|---|---|---|---|---|---|
| **R1 kda_hybrid** | **13.9 %** | **4.5 %** | 79.0 % | 89.2 % | 79.6 % | 82.60 % | 2.4559 | 0.3568 |
| R2 gdn_hybrid | 11.5 % | 2.5 % | 74.1 % | 87.9 % | 71.5 % | 77.82 % | 2.5082 | 0.4357 |
| **R3 kda_only** | 7.5 % | 1.6 % | 78.6 % | **90.1 %** | **82.3 %** | **83.64 %** | **2.4379** | **0.3328** |
| R4 gdn_only | 5.8 % | 0.1 % | 68.3 % | 87.5 % | 72.7 % | 76.17 % | 2.4623 | 0.3779 |

⚠️ *Relative score is `(student − chance)/(teacher − chance)`, computed here. Teacher: MMLU 0.6500, GSM8K 0.5353 strict, ARC-C 0.5922, HellaSwag 0.7816, WinoGrande 0.7048.*

### ✅ Mechanism: KDA beats GDN on **every single measurement**

Held-out CE, held-out KL, all three 0-shot benchmarks, MMLU, GSM8K — **seven for seven, in both layouts.** Held-out KL puts it at **+22.1 %** (hybrid) and **+13.6 %** (all-linear) for GDN. On MMLU the mechanism effect is 1.8 σ and 1.3 σ, small but the same sign. **This result is no longer in doubt within this experiment.**

### ⭐⭐⭐ Layout: not "free" — free on SHORT tasks, expensive on LONG ones

| | KDA hybrid → all-linear | GDN hybrid → all-linear |
|---|---|---|
| held-out CE (2048-token packed web text) | **better** (2.4559 → 2.4379) | **better** (2.5082 → 2.4623) |
| held-out KL | **better** (0.3568 → 0.3328) | **better** (0.4357 → 0.3779) |
| 0-shot mean relative | **+1.0 pts** | −1.7 pts |
| **MMLU, 5-shot** | **−2.56 pts raw — 4.7 σ** | **−2.29 pts raw — 4.3 σ** |
| **GSM8K strict** | 0.0243 → 0.0083 | 0.0136 → **0.0008** |

⭐⭐ **The two 5-shot tasks are the only ones where the layout matters, and there it matters a great deal — over 4 σ, replicated independently in both mechanisms.** A 5-shot prompt carries five worked exemplars that the model must still have access to when it reaches the real question. **Nine attention layers can look any of them up exactly; a fixed recurrent state carries a lossy summary.** The all-linear students lose roughly half the little MMLU headroom their hybrids had.

### ⭐⭐⭐ The corroborating signal: format compliance collapses with the layout

GSM8K's two filters are the same generations scored twice — `strict-match` requires the `####` marker the exemplars demonstrate five times, `flexible-extract` takes the last number regardless:

| | strict | flexible | **flexible / strict** |
|---|---|---|---|
| **R1 kda_hybrid** | 0.0243 | 0.0349 | **1.44×** |
| R2 gdn_hybrid | 0.0136 | 0.0235 | **1.73×** |
| R3 kda_only | 0.0083 | 0.0326 | **3.93×** |
| R4 gdn_only | **0.0008** | 0.0326 | **40.75×** |

⭐⭐⭐ **Monotone, and in exactly the order the layout predicts.** `gdn_only` reaches the right number about as often as `kda_only` does — and emits the required marker **once in 1,319 problems**. **This is not an arithmetic failure; it is a failure to reproduce a format demonstrated five times earlier in the prompt.** It is an entirely independent measurement of the same thing MMLU shows, on the same axis, from a different task.

### ⚠️⚠️ What this corrects

**Revision 24 said `kda_only` "costs nothing measurable" and treated that as the headline.** That was true of the evidence then — three 0-shot benchmarks, where it still holds — and it is **wrong as a general claim**. The honest version:

> Converting the last nine attention layers is free on short-context tasks and on held-out language modelling — it is in fact slightly *better* there — and costs about half the remaining MMLU headroom and most of the residual GSM8K format compliance. It buys **48.9×** inference state against the hybrid's **3.77×**. **Whether that trade is worth taking depends entirely on the context length you care about**, which is a far more useful sentence for the thesis than either "free" or "worse".

⭐ **And it reconciles every apparent disagreement in this document.** Held-out CE, held-out KL and the 0-shot benchmarks all measure short-range behaviour and all favour the all-linear layout slightly. MMLU and GSM8K measure retrieval across a long prompt and both favour the hybrid decisively. **They were never in conflict; they measure different regimes.**

📌 **A narrower version of the withdrawn "5-shot" idea survives here, and it is worth being precise about the difference.** Revision 20 claimed the *student* cannot use few-shot exemplars, and ARC-C at 5-shot falsified that — it improves. This is a **between-students** claim: given that all four can use exemplars, *the ones that keep attention use them better*. The ARC-C result does not bear on it.

⚠️ **All four students sit near the MMLU floor** — 5.8–13.9 % relative, all between 6 σ and 15 σ above chance. **The layout comparison is a comparison of small numbers**, and the 4 σ is on within-task sampling error, which says nothing about training-seed variance.

---

## ⭐⭐⭐ The loss components, all four — and **KL is where the mechanisms separate**

*Mean of the last twenty logged microbatches, same extraction for every run. At `epoch 0.7655` the model is still inside its first pass, so this data is effectively unseen — close to, but not the same as, the held-out split.*

| | CE | **KL** | `hs_cosine` | composite |
|---|---|---|---|---|
| **R1 kda_hybrid** | **2.2613** | **0.3144** | 0.0688 | **0.5564** |
| R3 kda_only | 2.2757 | 0.3342 | **0.0676** | 0.5625 |
| R2 gdn_hybrid | 2.2834 | 0.3545 | 0.0697 | 0.5694 |
| R4 gdn_only | 2.3031 | 0.3793 | 0.0745 | 0.5812 |
| *spread across the four* | *1.8 %* | ***20.6 %*** | *10.2 %* | |

⭐⭐ **KL carries almost all of the signal.** Cross-entropy spans 1.8 % across four architectures; **KL spans 20.6 %**. And KL is the *behavioural* metric — how well the student reproduces the teacher's distribution — and the one Arcee report.

| GDN − KDA, relative | CE | **KL** | `hs_cosine` |
|---|---|---|---|
| hybrid | +1.0 % | **+12.8 %** | +1.3 % |
| all-linear | +1.2 % | **+13.5 %** | +10.2 % |

⭐⭐⭐ **GDN's KL is ~13 % higher than KDA's in both layouts, to within 0.7 points.** That is the cleanest single statement the 2×2 produces: **replacing KDA's per-channel forget gate with GDN's per-head one costs about 13 % in token-averaged KL against the teacher, independently of how many layers you convert.** It is not an `hs_cosine` artefact — `hs_cosine` moves by 1.3 % in the hybrid.

**Four-way against Arcee's published ≈ 0.2** *(theirs, their corpus, their hybrid)*: kda_hybrid **0.314** · kda_only **0.334** · gdn_hybrid **0.355** · gdn_only **0.379**.

### ⚠️⚠️ This RETRACTS revision 24's explanation of the loss/benchmark disagreement

Revision 24 argued that held-out loss and benchmarks disagree on the layout axis *because the composite is 60 % `hs_cosine`, a representational measure*. **The decomposition does not support that.** Splitting KDA's layout gap by component:

| | contribution | share |
|---|---|---|
| CE | +0.00288 | +47 % |
| **KL** | **+0.00396** | **+65 %** |
| `hs_cosine` | **−0.00072** | **−12 %** |

⭐ **`hs_cosine` is the one component that goes the other way** — `kda_only` matches the teacher's hidden-state directions *better* than `kda_hybrid` does (0.0676 against 0.0688). The layout gap is driven by CE and KL, both behavioural. **The explanation I gave was wrong, and this is exactly the measurement that catches it.**

**The correct reading is duller and more defensible.** The layout costs something real but **small** — **+6.3 % KL for KDA, +7.0 % for GDN**, about half the mechanism effect. Three 0-shot benchmarks with standard errors of ±0.0146 (ARC-C) and ±0.0045 (HellaSwag) cannot resolve a difference that size. **It is not a contradiction between metric types; it is a small effect and a blunt instrument.**

⭐ **And the practical conclusion survives intact, on firmer ground.** The layout cost is real, measurable on distillation metrics, **and below the resolution of the benchmarks** — while buying **48.9×** inference state against **3.77×**. That is a better sentence for the thesis than "it costs nothing", because it is true.

✅ **RESOLVED rev 29** — measured on the held-out split ("Held-out CE, KL and `hs_cosine`, measured directly"). *Was:* ⚠️ **Still owed: held-out KL.** These are training-stream microbatches — rank 0, about 41 k tokens, fresh data but not the held-out split, and noisier than an eval pass. The ordering is consistent across all four and matches `eval_loss` exactly, so it is unlikely to be an artefact; **but the number that goes in the thesis should be measured on the held-out split for teacher and all four students in one process.**

---

## ⭐⭐⭐ The 2×2 on benchmarks — the independent test, and it changes one conclusion

*0-shot, frozen protocol, `use_cache: True`, full sets. 2026-09-28.*

| raw | ARC-C `acc_norm` | HellaSwag `acc_norm` | WinoGrande `acc` |
|---|---|---|---|
| teacher | 0.5922 ±0.0144 | 0.7816 ±0.0041 | 0.7048 ±0.0128 |
| **R1 kda_hybrid** | **0.5205** ±0.0146 | 0.7240 ±0.0045 | 0.6630 ±0.0133 |
| R2 gdn_hybrid | 0.5034 ±0.0146 | 0.7174 ±0.0045 | 0.6464 ±0.0134 |
| **R3 kda_only** | 0.5188 ±0.0146 | **0.7289** ±0.0044 | **0.6685** ±0.0132 |
| R4 gdn_only | 0.4838 ±0.0146 | 0.7153 ±0.0045 | 0.6488 ±0.0134 |

**Relative score** `(student − chance)/(teacher − chance)` — ⚠️ *computed here, not published*:

| | ARC-C | HellaSwag | WinoGrande | **mean** |
|---|---|---|---|---|
| **R3 kda_only** | 78.6 % | **90.1 %** | **82.3 %** | **83.64 %** |
| **R1 kda_hybrid** | **79.0 %** | 89.2 % | 79.6 % | **82.60 %** |
| R2 gdn_hybrid | 74.1 % | 87.9 % | 71.5 % | 77.82 % |
| R4 gdn_only | 68.3 % | 87.5 % | 72.7 % | 76.17 % |

### ✅ The mechanism result survives an independent metric

**KDA beats GDN in all six comparisons** — three benchmarks × two layouts — by **+4.78** points of mean relative score in the hybrid layout and **+7.47** all-linear.

⚠️ **How much confidence that buys, stated carefully.** Pooling the six with Stouffer's method gives **Z ≈ 3.1**. But those standard errors are **within-task sampling error only**; they say nothing about **training-seed variance**, which is the dominant unknown and which no run in this project has ever measured. **Read 3.1σ as "the direction is unlikely to be an accident of which questions are on the test", not as "the mechanisms differ by this much."** Individually only HellaSwag/all-linear clears 2σ (2.14).

### ⚠️⚠️ And the layout conclusion REVERSES

| effect, in points of mean relative score | |
|---|---|
| mechanism (KDA − GDN), hybrid | **+4.78** |
| mechanism (KDA − GDN), all-linear | **+7.47** |
| layout (all-linear − hybrid), KDA | **+1.04** |
| layout (all-linear − hybrid), GDN | **−1.65** |

**On held-out composite loss the all-linear students are clearly worse** (+0.0058 KDA, +0.0123 GDN). **On benchmarks the layout effect is indistinguishable from zero** and changes sign between mechanisms. Per benchmark, R3 against R1 reads −0.0017 / +0.0049 / +0.0055 — every one inside one standard error. **`kda_only` and `kda_hybrid` are indistinguishable on quality.**

⚠️⚠️ **SUPERSEDED — see "the layout effect depends on CONTEXT LENGTH" above.** This held for the three 0-shot benchmarks and still does; **MMLU and GSM8K, both 5-shot, show the hybrid ahead at over 4 σ.** The claim "costs nothing measurable" was true of the evidence available and is wrong in general.

⭐ `kda_only` carries a **48.9×** smaller inference state at 32k against the hybrid's **3.77×** (B9a) and costs nothing on three **short-context** benchmarks. ⭐ It also **independently reproduces Arcee's published finding** that their KDA-only student beats their hybrid on MMLU and ARC-C — arrived at on a different corpus, a different budget and our own code.

### ⚠️⚠️ RETRACTED — see the component decomposition above

*The explanation below was written before the per-component numbers existed, and the decomposition contradicts it: `hs_cosine` contributes **−12 %** of KDA's layout gap, i.e. it favours the all-linear student. The layout gap is driven by CE and KL. Kept, struck through in substance, because the reasoning is a good example of a plausible story that a measurement killed.*

### ~~Why the two metrics disagree, and why that is the interesting part~~

The held-out objective is `0.2·CE + 0.2·KL + 0.6·hs_cosine` — **60 % of it is `hs_cosine`, a measure of how well the student's hidden states point the same way as the teacher's.** That is *representational* similarity. Benchmarks measure *behaviour*.

**The mechanism axis moves both together; the layout axis pulls them apart.** Removing nine attention layers makes the student structurally further from the teacher's internals — there is simply less of the teacher's machinery left — while it answers questions just as well. **Matching the teacher's hidden states is not the same as matching its behaviour, and this 2×2 is where the two come apart.**

⚠️ *Superseded. The composite does rank layouts in the same direction as KL and CE; the benchmarks simply cannot resolve an effect that small.*

### ✅ RESOLVED rev 29 — ~~The hole: held-out KL has never been measured~~

⚠️ **KL is the behavioural half of the objective and the number Arcee report (≈ 0.2 against our R1's ≈ 0.35) — and it has not been compared across the four students.** DistillKit logs only the composite `eval_loss` on held-out data; the per-microbatch KL prints bounce too much to compare (R2: last-logged **0.3908** against a last-twenty mean of **0.3545**).

**This matters more than it looks.** If KDA wins on `hs_cosine` but ties on KL, the claim is *"KDA's internals resemble the teacher's more"*, not *"KDA behaves more like the teacher"* — and the benchmark result would be doing all the work. Two things close it:

1. **Free, now:** the last-twenty means from each training log (command in the run log above) — a train-side KL comparison at no cost.
2. **Proper:** held-out **CE and token-averaged KL at T = 2.0** for all four checkpoints plus the teacher, in one process. That produces the numbers the Results table already uses for R1 (CE 2.4559) and makes the Arcee comparison four-way.

---

## ⭐⭐⭐ The 2×2 is complete — and it replicates

*R3 (58650592) and R4 (58650595) finished 2026-09-27. All four cells: 300 M tokens, FineWeb-Edu, 4 nodes, `grad_accum: 2`, identical YAML but for the student and the output path.*

### Held-out combined loss, final

| | **KDA** | **GDN** | mechanism gap |
|---|---|---|---|
| **hybrid 3:1** (27 linear) | **0.5686** | 0.5814 | **+0.0128** |
| **all-linear** (36 linear) | **0.5744** | 0.5937 | **+0.0193** |
| *layout cost* | *+0.0058* | *+0.0123* | |

⭐⭐⭐ **The same strict ordering holds at all ten eval points in all four runs** — forty numbers, one rank order:

```
kda_hybrid  <  kda_only  <  gdn_hybrid  <  gdn_only
```

### Three things follow, and the third is the one to write down

⚠️ **Read this section against the benchmark section above, which was written later.** The loss differences here are **0.0128–0.0193 on a quantity near 0.57** — 2–3 % relative. The only noise estimate this project has is the **~0.023-nat seed spread** recorded for "the untrained hybrid", which propagates to **at least ~0.005** in the composite. **So the gap is roughly 2.5–4× a crude noise floor: suggestive, not conclusive, on its own.** The ten-point consistency *within* a run is weak evidence — a trajectory that starts behind tends to stay behind. **What carries the mechanism claim is the cross-layout replication and, above all, the benchmarks.**

**1. The mechanism result replicates in an independent layout.** R2 alone was one run with a consistent trajectory; R3/R4 repeat it with nine more converted layers and a differently-initialised student. **KDA is ahead in both cells and at every eval point.** Two independent layouts agreeing is a far stronger claim than ten points from one trajectory.

**2. ⭐ The gap scales with the number of converted layers.** Per linear layer it is **4.74e-4** in the hybrid (0.0128 / 27) and **5.36e-4** all-linear (0.0193 / 36) — within 13 % of each other. **The mechanism difference behaves like a per-layer property, not a whole-model artefact.** That is what a gating difference *should* look like, and it is the closest thing to a mechanistic confirmation this design can produce.

**3. ⭐⭐ Mechanism matters more than layout.** Averaged across cells the mechanism effect is **0.01605** and the layout effect **0.00905** — a ratio of **1.77×**. ⚠️ *On benchmarks this ratio is far larger still: the layout effect there is indistinguishable from zero.* Concretely: **`kda_only` beats `gdn_hybrid` at every single eval point**, so choosing KDA over GDN buys more than keeping nine full-attention layers does. Given that dropping those nine layers takes inference state from 3.77× to **48.9×** smaller (B9a), that is a consequential sentence.

### ⚠️ The layout is cheap, and that is its own finding

Converting the last nine attention layers costs KDA **0.0058** in held-out combined loss — about one third of what changing the mechanism costs — and buys a **13× further reduction in inference state**. GDN pays 2.1× more for the same move (0.0123), which is consistent with everything above: fewer attention layers means more of the model is the mechanism.

### Throughput — and the layout goes the *wrong* way

| tok/s | **KDA** | **GDN** | GDN advantage |
|---|---|---|---|
| hybrid | 25,455 | **29,082** | **1.14×** |
| all-linear | 23,081 | **27,402** | **1.19×** |

⭐ **GDN is faster in both layouts**, consistent with `_use_fused_qkv_conv` (predicted in `building_GDN_student_track_c.md` §2 before any of this ran).

⚠⚠ **And all-linear is SLOWER than the hybrid for both mechanisms** — 0.91× for KDA, 0.94× for GDN. That is not a defect: A6 measured a single KDA layer at **0.31× an attention layer's throughput at T = 2048**, with crossover near 16k. Replacing nine more attention layers at a 2,048-token training length therefore *costs* speed. **This is the model-level, measured confirmation that KDA's advantage is an inference property, not a training one** — the doc asserted it; these four runs measure it.

### Health — all four clean

| | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| `PREWARM params` | 4,999,026,076 | 5,141,828,536 | **5,046,996,688** | **5,237,399,968** |
| `Froze N tensors` | 147 ×16 | 147 ×16 | **147 ×16** | **147 ×16** |
| `roll back to CPU` | 0 | 0 | **0** | **0** |
| `train_runtime` count | 1 | 1 | **1** | **1** |
| `epoch` | 0.7655 | 0.7655 | **0.7655** | **0.7655** |
| held-out curve | monotone | monotone | **monotone** | **monotone** |
| `train_mem_gpu_peaked_delta` | — | 7.71 GB | **7.71 GB** | **7.73 GB** |

⭐ **The all-linear backward pass — flagged before launch as the one genuinely untested thing — worked in both mechanisms.** Neither `only` student had ever had a gradient computed through 36 consecutive linear layers.

📌 **Do not read the `train_mem_gpu_alloc_delta` spread** (10.11–10.73 GB). It has no monotone relationship to trainable-parameter count across the four cells, so it is dominated by activation buffers rather than optimizer state. The quantity that actually constrains you, `peaked_delta`, is **flat at 7.71–7.73 GB across all four**.

### ⚠️⚠️ Three caveats, none of which the numbers dissolve

**1. Held-out combined loss is 60 % `hs_cosine`.** `0.2·CE + 0.2·KL + 0.6·hs_cosine`, so "KDA is better" here means chiefly *KDA's hidden states track the teacher's direction better*. ✅ *(done rev 28–29)* **The interpretable number is held-out CE**, and R1's is the only one measured (2.4559). **Run `test_r1_ce.py` against all four checkpoints** before any of this goes in the thesis — one debug job, and it converts a composite training objective into the quantity the results table already uses.

**2. ⚠️ Do NOT compare the last logged `distillation_loss/*` across runs.** It is a single rank-0 microbatch and it bounces: R2's last-logged CE was **2.119** while its mean over the last twenty was **2.283**. R3 and R4 last-logged 2.107 and 2.111, which would wrongly suggest the `only` students beat the hybrids — the held-out curve says the opposite. **`eval_loss` is the cross-run quantity; the per-microbatch prints are not.**

**3. n = 1 per cell.** Four runs, one seed each. The replication across layouts is real evidence; it is not a variance estimate. Two more seeds in the hybrid row would settle it and cost ~6.5 h. ✅ *(rev 42: seeds 43 and 44 for R1 and R2 launched 2026-10-08.)*

⚠️ **And none of these four can move open decision 2.** All share R1's corpus and frozen MLPs, so all four are expected at ~14 % MMLU and ~3 % GSM8K. **This cross measures mechanism and layout; it does not recover the missing capability.**

*Total cost of the 2×2: 12.8 h wall-clock, ~1,637 local hours — about 1.5 % of the remaining budget.*

---

## ⭐⭐ R2 — the mechanism comparison, first result

*Job 58650590, finished 2026-09-26. `afm_gdn_hybrid31_init` at `8f40624`, `"dirty": false`, every YAML setting identical to R1's but the student and the output path.*

### Every pre-registered check passed

| | expected | measured |
|---|---|---|
| `PREWARM params` | 5,141,828,536 | **5,141,828,536** ✅ |
| `PREWARM class` | `AFMKDAForCausalLM` | ✅ |
| `Froze N tensors` | **147**, ×16 | **147**, ×16 ✅ |
| step-0 `cross_entropy` | ≈ 7.9 | **7.981** ✅ (≈ 2.2 would mean the teacher loaded) |
| `roll back to CPU` | 0 | **0** ✅ |
| `train_runtime` | exactly 1 | **1** ✅ — the job did not split |
| `'loss':` lines | 458 | **458** ✅ = 4,580 steps at `logging_steps: 10` |
| `epoch` | 0.7655 | **0.7655** ✅ — identical to R1 |
| NaN / OOM / traceback | none | **none** |

**Token count confirmed two ways**, as the doc requires: `14.2 samples/s × 10,320 s = 146,544` sequences × 2,048 = **300.1 M**, against 4,580 × 32 × 2,048 = 300,154,880 configured.

### ⭐⭐ KDA against GDN, everything else held fixed

| | **R1 — KDA hybrid** | **R2 — GDN hybrid** | |
|---|---|---|---|
| parameters | 4,999,026,076 | **5,141,828,536** | GDN +2.9 % |
| trainable | 945 M in 477 tensors | **1,089 M in 396 tensors** | GDN +15 % |
| wall-clock | 11,790 s = 3.27 h | **10,320 s = 2.87 h** | |
| **throughput** | 25,455 tok/s | **29,082 tok/s** | ⭐ **GDN 1.14× faster** |
| `train_mem_gpu_alloc_delta` | 10.73 GB | **10.36 GB** | GDN −3.4 % |
| step-0 CE / KL / `hs_cosine` | 7.918 / 10.13 / 0.841 | 7.981 / 9.349 / **0.8404** | |
| end CE / KL / `hs_cosine` | 2.263 / 0.353 / 0.0705 | 2.283 / 0.355 / 0.0697 | indistinguishable |
| **held-out combined, final** | **0.5686** | **0.5814** | ⭐⭐ **KDA ahead by 0.0128** |

**The held-out curve, all ten eval points:**

| eval | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| **R1 KDA** | 0.7429 | 0.6954 | 0.6549 | 0.6308 | 0.6109 | 0.5923 | 0.5806 | 0.5721 | 0.5688 | **0.5686** |
| **R2 GDN** | 0.7695 | 0.7225 | 0.6792 | 0.6537 | 0.6297 | 0.6080 | 0.5936 | 0.5849 | 0.5817 | **0.5814** |

⭐⭐ **KDA is ahead at every one of the ten points, and the gap narrows monotonically from 0.0266 to 0.0128.** Both curves fall monotonically with no overfitting anywhere.

⭐ **And KDA wins from behind on parameters.** GDN carries 143 M more parameters and 15 % more trainable ones, and still ends higher. Held-out loss per parameter is not the comparison anyone asked for, but it makes the direction harder to dismiss.

### What this does and does not establish

⭐⭐ **This is the first evidence on KDA's own central claim**, in the setting its authors did not test. The Kimi Delta Attention paper stakes itself on a **per-channel** forget gate (2,560 decay values per token, through a 128-dim bottleneck) against Gated DeltaNet's **per-head** one (20 scalars). C1 established that the two students start **0.0022 nats apart**, an order of magnitude inside seed noise, so the starting line is not the explanation. After 300 M tokens of identical training on an identical corpus, **KDA is ahead by 0.0128 in held-out combined loss.**

⚠️ **But it is n = 1 per cell, one seed, and the ten eval points are one trajectory, not ten independent samples.** Consistency across the ten is what makes it more than a coin flip; it is *not* a significance test. A second seed per cell would settle it and costs two more runs. ✅ *(rev 42: seeds 43 and 44 for R1 and R2 launched.)*

⭐ **The throughput result is the surprise, and the doc called it in advance.** `claude/building_GDN_student_track_c.md` §2 noted that GDN's `_use_fused_qkv_conv` fast path fires under exactly our training conditions — no mask, `use_cache=False`, no `cu_seqlens` — collapsing three short convolutions into one `causal_conv1d`, *"which is worth knowing if a throughput comparison ever looks surprising."* It looked surprising, and that is the candidate explanation: **14 % faster and 3.4 % less GPU memory on 15 % more trainable parameters.** 📌 Offered as the mechanism, not proven as one — isolating it would need the path disabled and the run repeated.

⚠️ **So the honest framing is a trade, not a win.** *KDA's fine-grained gating buys a small held-out-quality advantage at a ~14 % throughput cost and ~3 % more memory.* Both halves belong in the thesis, and the benchmark numbers are still to come.

📌 **One difference worth recording, not worrying about.** R2's first logged `grad_norm` is **11.15** against R1's 6.36, falling to 2.558 then 1.461 within thirty steps and to 0.071 by the end. `max_grad_norm: 0.5` clips throughout and nothing destabilised. Consistent with 15 % more trainable parameters and with GDN's `A_log ~ log(uniform(0, 16))` against KDA's `uniform(1, 16)` — GDN can draw heads with near-zero decay, i.e. near-unbounded memory, at initialisation. *Inference, not measurement.*

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

## ⚠️⚠️ R5's first launch died in dataset preprocessing — the 600-second barrier

*Job 59074838, 2026-10-01. Relaunched as **59234295** on 2026-10-02 after the fix below.*

### What happened

The log carries the stack trace of the collective that timed out, so none of this is inferred:

```
#0 barrier            torch/distributed/distributed_c10d.py:5230
#2 wait_for_everyone  accelerate/state.py:403
#3 _goes_first        accelerate/state.py:409
#4 main_process_first accelerate/state.py:526
#6 _prepare_dataset   trl/trainer/sft_trainer.py:894
#7 __init__           trl/trainer/sft_trainer.py:805
#8 __init__           distillkit/trainer.py:35
```

⭐ **TRL has exactly one process tokenize, and everyone else waits.** `SFTTrainer._prepare_dataset` wraps the map in `accelerate.main_process_first()`; global rank 0 does the work, ranks 1–15 sit in a NCCL `barrier()` whose watchdog aborts the process at **600,000 ms**. Every rank then self-aborted — *"to avoid data inconsistency, we are taking the entire process down"* — and all four `srun` tasks exited 1. *(DistillKit's own `do_distill` wraps `load_data` in a second `main_process_first`; either one is enough to produce this.)*

### ⭐⭐⭐ It missed by one percent

| | |
|---|---|
| documents | **2,921,000** |
| sustained rate, `num_proc=16` | **4,819 / s** — measured at 4:31 and again at 5:33, dead flat |
| projected completion | **606 s** |
| watchdog limit | **600 s** |
| rate that would have sufficed | **4,868 / s** |

It was at 55 % when the barrier blew. ⚠️ **R1–R4 were never at risk**: `num_samples: 500000` is 106 s of the same work. **The single change that made R5 a one-epoch run is the change that killed it.**

### ⭐ The sibling ranks were the difference

| | rate | total |
|---|---|---|
| 4-node run — rank 0 sharing its node with 3 sibling ranks | 4,819 ex/s | **606 s** ❌ |
| single-process warm-up — rank 0 alone on the node | **6,150 ex/s** | **474 s** ✅ |

Same `num_proc=16`, same data, **28 % faster alone.** The three siblings were doing nothing except spinning in the barrier and stealing CPU from the one process that had work. ⭐ **A barrier is not free for the process it is waiting on.**

### ✅ The fix: warm the `datasets` map cache in a single-process job

`datasets.map` caches by fingerprint. Letting rank 0 *finish* the map once, somewhere with no barrier, makes every later launch skip it. Run the real YAML at `--num_processes 1` with `output_path` redirected to a throwaway directory. Verified by re-running: the second launch went from loading the teacher straight to `Starting training.` with **no `Tokenizing train dataset` line and no `Packing train dataset` line.**

⚠️ **The cache does NOT live in `HF_DATASETS_CACHE`.** For a dataset loaded with `load_from_disk`, `map` writes beside the dataset's own arrow files — `$WORK_LUKAS/datasets/fineweb_edu_3B_train/cache-*.arrow`. `$HF_HOME/datasets` stayed empty and that is correct.

| fingerprint | size | what |
|---|---|---|
| `cache-2b161db1…_of_00016` | 16 × 1.605 GB = **25.7 GB** | tokenize |
| `cache-9eb68d25…_of_00016` | 16 × 0.572 GB = **9.15 GB** | packed 2048-token sequences |
| `cache-73d1e91c…_of_00016` | 16 × 0.872 GB = 14 GB | ⚠️ **orphan** from the failed run — 872/1605 = **54 %**, exactly where it died |

⚠️ **`dataset_num_proc` is part of the cache key.** `map` with `num_proc=16` writes `cache-<hash>_00000_of_00016.arrow`; asking for 32 wants `_of_00032`, which is a cache miss and a full re-tokenize. **So raising `num_proc` and warming the cache are alternative strategies, not complementary ones.** The cache is the better one: measured rather than extrapolated, and it pays off again on every resume.

### ⭐⭐⭐ And it measured the epoch exactly

Re-running the warm-up with `max_steps: -1` and `num_train_epochs: 1` makes the trainer print the real total:

```
559,146 steps at 1 process (per_device 1 x grad_accum 2 = 2 sequences/step)
  -> 1,118,292 packed sequences of 2048
  -> 2,290,262,016 tokens  =  2.2903 B  =  ONE EPOCH
  -> 784.1 effective tokens per document   (the estimate was 774, off by 1.3 %)
  -> 34,946.6 steps at global batch 32
```

⭐ Corroborated independently by the pack cache's own size: `1,118,292 × 2048 × 4 bytes = 8.53 GiB` against 9.15 GiB on disk.

**R5 runs at `max_steps: 34940`** — 99.98 % of an epoch, and `3494 × 10 = 34940` and `1747 × 20 = 34940` both land exactly, so ten checkpoints and twenty evals divide cleanly with every second eval coinciding with a save.

⚠️⚠️ **Why not `num_train_epochs: 1.0`?** Because the derived total is 34,946 *or* 34,947 depending on how `DistributedSampler` pads 1,118,292 across 16 ranks, and with `save_steps: 3494` an eleventh checkpoint would make `save_total_limit: 10` **rotate away `checkpoint-3494`** — the earliest point on the quality-vs-tokens curve, which is this run's deliverable. R1 already lost that curve once to `save_total_limit: 1`. Three further reasons: the token budget is the invariant and "one epoch" is a property of a corpus that will change; `num_train_epochs` makes the step count and therefore the LR schedule depend on the node count, so a resume on 3 nodes silently retunes the cosine; and `r5_resume.sbatch` greps `max_steps` to compute remaining work.

### The other two defects the failure exposed

⚠️ **`$OUTDIR` is created before anything useful happens.** `do_distill` opens with `os.makedirs(config.output_path, exist_ok=True)` and writes `distillkit_config.yaml`. The sbatch's trailing stamp block then ran *after* `srun` failed, because the script has no `set -e`. So a failed launch leaves a populated `$OUTDIR` and the `[ -e "$OUTDIR" ]` guard refuses every resubmission. **Fix: capture `SRUN_RC=$?` on the line immediately after the srun block and exit before stamping.**

✅ **`r5_resume.sbatch` written** (`claude/r5_resume.sbatch`). It **inverts** the existence guard (a resume requires the directory and a checkpoint), **keeps** the `output_path == OUTDIR` guard unchanged, requires `trainer_state.json` — the exact file whose absence killed the 09-29 resume test — warns if `global_step*/` is missing (weights reload, **Adam resets**), and injects `resume_from_checkpoint` into a derived YAML with the `yaml` module rather than by hand. Verified against `main.py:380`: `config.training_args.get("resume_from_checkpoint", None)`, so the key belongs **under `training_args`**. ⚠️ It has never been exercised.

---

## ⭐⭐ R5 finished — the 0-shot result, and the pre-registration for everything else

*⚠️ Rev 42: checkpoints 3494–31446 were deleted on 2026-10-08 to free disk; their probe results are kept in `eval_results/arith_probe_v2`. Job 59234295. `output/r5_stage1_kda_hybrid31_1ep/` held `checkpoint-3494` … `checkpoint-34940` (ten, evenly spaced — the `max_steps: 34940` / `save_steps: 3494` / `save_total_limit: 10` arithmetic worked exactly) plus the final model at the top level, written by `trainer.save_model()` after step 34,940.*

### The 0-shot benchmarks — and they did not move

*Frozen protocol, `pretrained=output/r5_stage1_kda_hybrid31_1ep` (top level = final weights, the same step as `checkpoint-34940`), `use_cache: True`, bf16.*

| | teacher | **R1** (300 M) | **R5** (2.29 B) | R5 − R1 | ~95 % CI, unpaired | rel. R1 | **rel. R5** |
|---|---|---|---|---|---|---|---|
| ARC-C `acc` | 0.5563 | 0.4778 | **0.4642** | −0.0136 | [−0.054, +0.027] | 74.4 % | 69.9 % |
| **ARC-C `acc_norm`** | 0.5922 | 0.5205 | **0.5000** | **−0.0205** | [−0.061, +0.020] | 79.0 % | **73.1 %** |
| HellaSwag `acc` | 0.5882 | 0.5449 | **0.5413** | −0.0036 | [−0.017, +0.010] | 87.2 % | 86.1 % |
| **HellaSwag `acc_norm`** | 0.7816 | 0.7240 | **0.7220** | −0.0020 | [−0.015, +0.011] | 89.2 % | **88.8 %** |
| **WinoGrande `acc`** | 0.7048 | 0.6630 | **0.6464** | **−0.0166** | [−0.054, +0.021] | 79.6 % | **71.5 %** |
| *0-shot mean relative* | | | | | | *82.6 %* | ***77.8 %*** |

⭐⭐ **Seven and a half times the tokens, and every metric is lower.** Read it carefully:

- **"Better" is ruled out.** The upper ends of the intervals are +2.0, +1.1 and +2.1 points. **Whatever 2 B further tokens of FineWeb-Edu bought, it was at most a point or two on any 0-shot benchmark, and most likely nothing.**
- **"Worse" is NOT established.** No difference clears 1 σ unpaired (z = −0.99, −0.31, −0.88). Five-for-five in the same direction is weaker than it looks: `acc` and `acc_norm` are the same questions, so it is three tasks, and 3/3 by chance is p = 0.125. ⭐ **A paired test would be far tighter** — same questions, both models — but only if the 0-shot runs logged per-item samples.
- ⚠️ **In items, it is ~24 more wrong on ARC-C, ~21 on WinoGrande, ~20 on HellaSwag.** Small, consistent, unproven.

### ⭐ R5 on MMLU and GSM8K (added rev 43, jobs `eval_mml` / `eval_gsm`)

*Same frozen protocol as R1 (MMLU 5-shot `acc`, micro-average over 14,042 items; GSM8K 5-shot, 1,319 items), R5 top level (= `checkpoint-34940`).*

| | teacher | **R1** (300 M) | **R5** (2.29 B) | R5 − R1 | z (unpaired) | ~95 % CI on Δ | rel. R1 | **rel. R5** |
|---|---|---|---|---|---|---|---|---|
| **MMLU** `acc` | 0.6500 ±0.0037 | 0.3055 ±0.0039 | **0.3068** ±0.0039 | +0.0013 | 0.2 | [−0.010, +0.012] | 13.9 % | **14.2 %** |
| **GSM8K** strict | 0.5353 ±0.0137 | 0.0243 ±0.0042 | **0.0409** ±0.0055 | **+0.0166** | **2.4** | [+0.003, +0.030] | 4.5 % | **7.6 %** |
| GSM8K flexible | 0.5345 | 0.0349 | 0.0447 ±0.0057 | +0.0098 | | | | |
| *flexible / strict* | *1.00* | *1.44* | ***1.09*** | | | | | |

R5 MMLU by category (raw `acc`): humanities 0.2721 ±0.0065 · other 0.3312 ±0.0085 · social sciences 0.3396 ±0.0085 · STEM 0.3026 ±0.0082.

⭐⭐ **MMLU did not move by a single point with 7.6× the tokens.** The interval excludes any gain above ~1.2 raw points. Revision 27 explained MMLU by three things — 23 % of Arcee's tokens, no stage with unfrozen MLPs, a narrower corpus. **R5 removes the first, at least while the MLPs are frozen**: more stage-1 tokens on FineWeb-Edu are not the lever for MMLU. What remains is the frozen MLPs (**tested by stage 2**) and the corpus.

⭐ **GSM8K moved a little, and in the direction the arithmetic probe predicts.** +0.0166 strict (≈ 54 vs 32 problems solved), z ≈ 2.4. R5's pooled arithmetic is 1.20× R1's (45.9 vs 38.3); with revision 32's k = 2.28 effective operations per problem, 1.20^2.28 = 1.51 predicts 0.0243 × 1.51 = **0.037** against **0.041** measured. ⚠️ *Heuristic: k was fitted with probe v1, the ratio comes from probe v2.* It is still the GSM8K-follows-arithmetic account, now on a second model.

📌 **R5 keeps the GSM8K answer format better than R1** (flexible/strict 1.09 vs 1.44, teacher 1.00): when it gets the number, it writes `####`. Consistent with layout being the main driver of format loss (R5 is a hybrid like R1), and with format improving as the model improves at all.

⚠️ **R5 is still not a base model** (rev 37: latent, input-triggered damage). These numbers say what 7.6× tokens did to two benchmarks; they do not rehabilitate R5.

### ⚠️⚠️ R5 is not "R1 with more tokens" — the schedule stretched with it

R1's cosine ran over **4,580** steps; R5's over **34,940**. So R5 differs from R1 in **two** things, the token count and the shape of the learning-rate schedule, and this comparison cannot separate them. ⚠️ **The same applies to R5's own checkpoints**: `checkpoint-3494` is at ~229 M tokens but at a learning rate still near its 1e-3 peak — **it is an un-annealed model, not "R1 at 229 M tokens".** A curve across R5's checkpoints conflates tokens with annealing.

What the comparison *can* support: **neither more tokens nor a longer schedule, nor both together, improved 0-shot performance.**

### ⭐⭐⭐ Pre-registration — written 2026-10-06, before any of these is measured

⚠️ **Honest timing.** Written *after* the 0-shot results above were seen, and they plausibly correlate with what follows. Written *before* R5's held-out `eval_loss` has been read, before `test_r1_ce_math.py` has run on R5, and before any model has been through `arith_probe.py`. **Thresholds are copied into `arith_compare.py` as constants; that script prints the verdicts.**

#### A. Held-out `eval_loss` — from R5's own log

DistillKit's held-out set is `shuffle(seed=42).select(200)` of the same held-out split in R1 and R5, so **R5's `eval_loss` is on the identical 200 documents as R1's 0.5686** — a like-for-like number.

| | prediction |
|---|---|
| R5 final `eval_loss` | **below R1's 0.5686**, around 0.52–0.55. The objective keeps improving on its own distribution. |
| the eval curve | still falling at the last few evals, flattening as the LR reaches its floor |

⭐ **If A holds while the benchmarks above did not move, the distillation objective and the benchmarks have DECOUPLED beyond 300 M tokens** — the student gets better at matching the teacher on FineWeb-Edu and no better at anything measured off it. That is the corpus hypothesis in its sharpest form.

#### B. Where the improvement went — `test_r1_ce_math.py` on R5

R1's paired student−teacher gaps: **web +0.1595 · maths +0.3372 · instruction/code (Tulu 3) +0.3320**.

| | prediction for R5 |
|---|---|
| **web gap** (training distribution) | **shrinks** below +0.1595 by more than two paired SE (~0.014) |
| **maths and Tulu 3 gaps** (off-distribution) | **do not improve**: each stays above R1's value minus 0.02 |
| ratio off-distribution / web | **rises** above R1's 2.11× / 2.08× |

⭐ That is **specialisation**: more training on corpus X makes the student match the teacher better on X and not elsewhere. If instead all three gaps shrink proportionally, the 0-shot flatness needs a different explanation and this document's corpus hypothesis is weakened.

#### C. The arithmetic probe v2 — `arith_probe.py` / `arith_compare.py`

*Fifteen conditions, 2,286 problems per model, identical prompts for every model, 4-shot, greedy, `max_new_tokens=12`, one shared tokenizer. Teacher, R1–R5, and R5's ten checkpoints. Tiny problem spaces (`copy_1`, `copy_2`, `add_1`, `sub_1`, `mul_1`) are tested exhaustively rather than sampled.*

⚠️ **This is a new probe, not a re-run of the old one.** The old probe's operand ranges and op mix are not recorded, so its 76 % / 20 % are not comparable to anything below. **Every model, R1 and the teacher included, is re-measured here.**

| | test | threshold | **prediction** | if it fails |
|---|---|---|---|---|
| **P0** validity | teacher, `copy_1..3` pooled | ≥ 95 % | ✅ holds | the prompt format itself is broken — stop |
| **P1** reading | every student, `copy_1..3` pooled (n = 291) | ≥ 95 % | ✅ **students read fine** | ⚠️⚠️ **"the arithmetic circuit is destroyed" is wrong** — the damage is exact in-context retrieval, and stage 2 should target recall |
| **P2** digit sweep | student/teacher ratio, `add_4` vs `add_2` | falls by ≥ 0.15 | ✅ **falls** — the student loses more as more positions must align | damage does not scale with precision demanded; the positional story weakens |
| **P3** carry | student's (no-carry − carry) penalty minus the teacher's, `add_2..4` | ≥ 10 points | ✅ **carry hurts the student more** | the failure is not in carry propagation specifically |
| **P4** mechanism | KDA − GDN, pooled arithmetic, paired 95 % CI, both layouts | CI > 0 | ✅ **KDA ahead in both** — the v1 probe's ~2.5× | the largest mechanism effect in the project was a probe artefact |
| **P5** ⭐ **R5 vs R1** | pooled arithmetic, paired 95 % CI on R5 − R1, gap closure G | see bands | ⭐ **NULL**, with **EROSION** the next most likely | — |
| **P6** curve | R5's ten checkpoints, max − min of pooled arithmetic | < 5 points | ✅ **flat** | ⚠️ a rise is **uninterpretable** here — it conflates tokens with LR annealing |
| identity | `r5_ck34940` vs R5 top level | 100 % identical predictions | ✅ | the two "identical" weight files differ — find out why |

**P5 bands**, applied mechanically by `arith_compare.py` to the paired 95 % CI [lo, hi] on R5 − R1, in points:

| verdict | rule | meaning |
|---|---|---|
| **RECOVERY** | lo > 0 **and** G ≥ 0.25 | the token budget **was** binding for arithmetic |
| SMALL GAIN | lo > 0, G < 0.25 | real but under a quarter of the gap; the corpus is still binding |
| **NULL** ⭐ *predicted* | −5 ≤ lo **and** hi ≤ +5 | **equivalence**, not just "CI includes zero": no change larger than 5 points. **The corpus is binding, not the budget.** |
| **EROSION** | hi < 0 | longer training on FineWeb-Edu made arithmetic **worse** |
| inconclusive | CI includes 0 but reaches past ±5 | cannot call |

⚠️ **Tokenisation can change what P2 means.** If AFM's tokenizer packs 1–3 digits into one token, there is no per-digit alignment inside the prompt for d ≤ 3, and the sweep measures vocabulary, not position. `arith_probe.py` prints how numbers of each length tokenize, and `arith_compare.py` flags it. **Read that block before reading P2.**

### What each outcome would do to stage 2

| outcome | consequence |
|---|---|
| **P1 fails** (students cannot copy) | the diagnosis is **retrieval**, not computation. Stage 2 must target exact recall — unfreezing the MLPs alone does not address it |
| P1 holds, **P5 NULL or EROSION** | **the corpus is binding.** S2-B (frozen MLPs + a few % synthetic arithmetic) is the indicated lever; S2-A (MLPs unfrozen, `lr 1e-5`, `hs_cosine → 0`) tests whether reorganisation helps on its own |
| P1 holds, **P5 RECOVERY** | **the budget is binding**, and a longer stage 1 is a lever after all — which contradicts the 0-shot result and would need explaining |
| **B confirms specialisation** | independent support for the corpus reading, from language modelling rather than benchmarks |

### How to run it

```bash
# summarise R5's training log -- one screen, the run-log quantities by the R1-R4 method
cd $WORK_LUKAS/distillkit
OUT=logs/r5_stage1_kda_hybrid31_1ep-59234295.out
python $WORK_LUKAS/afm_kda/summarize_run.py $OUT ${OUT%.out}.err --epoch-tokens 2290262016 --max-steps 34940

# the probe: free dry run, then a smoke on R1, then the four real jobs
cd $WORK_LUKAS/afm_kda
python arith_probe.py --dry-run | head -40
sbatch --export=ALL,SET=a arith_probe.sbatch      # teacher, R1, R2
sbatch --export=ALL,SET=b arith_probe.sbatch      # R3, R4, R5
sbatch --export=ALL,SET=curve1 arith_probe.sbatch # R5 checkpoints 1-5
sbatch --export=ALL,SET=curve2 arith_probe.sbatch # R5 checkpoints 6-10
python arith_compare.py $WORK_LUKAS/eval_results/arith_probe_v2
```

---

## ⚠️⚠️⚠️ R5 diverged during training — and what the probes found

*`summarize_run.py` on job 59234295, `test_r1_ce_math.py` on R5, and `arith_probe.py` v2 on teacher, R1–R5 and R5's ten checkpoints. 2026-10-07.*

### The run was not clean

Health checks all passed — 147 frozen ×16, one `train_runtime`, no tokenize pass (cache hit), no NaN, no traceback, tokens 1,118,103 measured against 1,118,080 configured. **And the run diverged anyway**, which no check in the table was built to catch:

| eval | step | LR | **held-out `eval_loss`** | | R5 checkpoint arithmetic (pooled) | copy |
|---|---|---|---|---|---|---|
| 1 | 1,747 | 1.0e-3 | **2.051** | ⚠️ spike | | |
| 2 | 3,494 | 9.9e-4 | 0.698 | | 16.5 % | 100 % |
| 3 | 5,241 | 9.7e-4 | 0.638 | best so far | | |
| 4 | 6,988 | 9.2e-4 | **1.742** | ⚠️⚠️ collapse | **0.8 %** | **0–2 %** |
| 5 | 8,735 | 8.8e-4 | 1.062 | | | |
| 6 | 10,482 | 8.2e-4 | **1.755** | ⚠️⚠️ collapse | **0.8 %** | **0–1 %** |
| 7 | 12,229 | 7.6e-4 | 0.963 | `grad_norm` **83,070** at ~12,480 | | |
| 8 | 13,976 | 6.8e-4 | 0.863 | recovering | 13.7 % | 100 % |
| 10 | 17,470 | 5.3e-4 | 0.650 | brief spike in the training stream just after | 28.7 % | 100 % |
| 12 | 20,964 | 3.7e-4 | 0.622 | clean from here | 35.1 % | 100 % |
| 16 | 27,952 | | 0.582 | | 44.4 % | 100 % |
| 20 | 34,940 | 1.0e-5 | **0.5715** | | **45.9 %** | 100 % |

The training-stream components tell the same story: healthy at epoch 0.10 (CE 2.41, KL 0.58, `hs_cosine` 0.098 — on R1's trajectory), then **back to CE 4.76 / KL 3.96 / `hs_cosine` 0.33 at epoch 0.15**, i.e. most of the way back to initialisation, and not clean again until epoch 0.55.

⭐⭐ **At checkpoints 6988 and 10482 the model could not copy a three-digit number.** That is not degraded arithmetic; it is a broken model. The probe caught the divergence in behaviour independently of the loss curve.

#### ⭐ Every spike happened while the LR was high — and R5 held it high for 7.6× longer

| | steps with LR > 5e-4 | steps with LR > 8e-4 |
|---|---|---|
| R1 (cosine over 4,580) | 2,304 | 1,348 |
| **R5 (cosine over 34,940)** | **17,580** | **10,285** |

All instability falls in R5's first ~55 %, while the LR was above ~5e-4; none after. R1 spent too few steps up there to meet it. **The recipe's `learning_rate: 1.0e-3` was tuned for a 300 M-token stage and is not safe held for long.** `max_grad_norm: 0.5` did not prevent it — clipping the gradient barely constrains an Adam step, whose size is set by the LR. ⚠️ *The mechanism (Adam second-moment collapse, a KDA gate parameter drifting, bf16) is not identified; the correlation with LR exposure is measured.*

**Consequences:**
- ⚠️⚠️ **R5 is not a clean "more tokens" experiment.** It is "more tokens, a longer schedule, and two collapses with recovery". The final model is effectively a recovered state plus ~1.4 B tokens of clean, annealing training.
- ⚠️ **Pre-registration A failed**: final `eval_loss` **0.5715 > 0.5686**. Recorded as failed; the instability is the obvious confound, but the prediction was unconditional.
- ⭐ **Any longer run — a clean R5′ or a stage 2 — needs a lower peak LR** (3e-4 to 5e-4) and a spike check in `summarize_run.py`, which now reports max `grad_norm` and the number of eval rises. ⚠️ Lowering the LR changes a second variable relative to R1; that is the price of a stable long run.

### ⚠️⚠️ `test_r1_ce_math.py` on R5 contradicts DistillKit's eval — unresolved

| | teacher | **R5** | gap | paired se | ppl ratio | *R1's gap* | *R1's se* |
|---|---|---|---|---|---|---|---|
| web (FineWeb-Edu held-out) | 2.2963 | **2.7632** | **+0.4669** | **0.152** | 1.60× | +0.1595 | 0.007 |
| maths (MetaMathQA) | 0.8673 | 1.4323 | +0.5650 | 0.157 | 1.76× | +0.3372 | 0.006 |
| instr/code (Tulu 3) | 1.7218 | **3.2525** | **+1.5307** | **0.380** | **4.62×** | +0.3320 | 0.010 |

The teacher row reproduces exactly, so the pipeline is the same one. ⚠️ **But DistillKit's own eval puts R5 level with R1** (0.5715 vs 0.5686): a CE penalty of +0.31 nats would add ~+0.06 to that composite, and it is not there. Its training-stream CE ended *better* than R1's (2.225 vs 2.261). ⭐ **The paired standard error is twenty times R1's** — a per-sequence standard deviation of ~1.2 nats against ~0.06. **That is a handful of catastrophic sequences, not a uniformly worse model.**

Three differences between the two measurements could matter: `sdpa` (this script) against `flash_attention_2` (DistillKit); plain concatenation against TRL's EOS-separated packing; and different documents. **A recovered-from-divergence model with a few parameters in a fragile regime, failing only on some inputs or one kernel, would produce exactly this.** ✅ *(ran — rev 37)* `ce_outliers.py` measures per-sequence and per-position CE, sdpa against FA2, per-layer hidden-state magnitude on the worst sequence, and weight drift R5 vs R1 — one debug job.

⚠️ **Pre-registration B** predicted the web gap would *shrink*. As measured it **tripled**, so B is recorded as **failed** — and as **not interpretable** until the outliers are explained, because the same measurement disagrees with the run's own eval.

📌 *The script's printed "validity check" compares against R1's reference gap; for any other student that line is meaningless. The teacher row is the validity check.*

### ⭐⭐⭐ The arithmetic probe v2

*15 conditions, 2,286 problems per model, identical prompts, 4-shot, greedy. Pooled = all 1,795 non-copy problems.*

| | teacher | R1 kda_hyb | R2 gdn_hyb | R3 kda_only | R4 gdn_only | **R5** |
|---|---|---|---|---|---|---|
| copy_1..4 | 100 | 100 | 100 / 98 at 4d | 100 | 100 | 100 |
| add_1 / sub_1 / mul_1 | 100 / 98 / 100 | 100 / 98 / 100 | 97 / 93 / 96 | 100 / 100 / 100 | 100 / 90 / 96 | 100 / 100 / 99 |
| **add_2** | 100 | 77.0 | 36.5 | 75.0 | 42.5 | **84.0** |
| **add_3** | 94.0 | 9.5 | 0.5 | 6.0 | 3.5 | **16.5** |
| add_4 | 79.0 | 0 | 0 | 0 | 0 | 0 |
| **sub_2** | 100 | 55.0 | 19.5 | 65.0 | 32.0 | **90.0** |
| **sub_3** | 98.5 | 3.0 | 0.5 | 7.5 | 3.0 | **14.5** |
| sub_4 | 80.0 | 0 | 0.5 | 0 | 0 | 0.5 |
| mul_2x1 | 100 | 84.0 | 59.0 | 78.5 | 68.5 | 86.5 |
| mul_2 | 68.0 | 18.0 | 11.5 | 17.5 | 14.0 | 23.0 |
| **pooled** | **91.0** | **38.3** | **24.7** | **38.7** | **28.7** | **45.9** |

#### ⭐⭐ The tokenizer changes the story

```
1-digit ' 6'     -> ['Ġ', '6']          one number token
2-digit ' 56'    -> ['Ġ', '56']         one number token
3-digit ' 513'   -> ['Ġ', '513']        one number token
4-digit ' 3357'  -> ['Ġ', '335', '7']   split LEFT-to-right, misaligned with place value
```

⚠️⚠️ **Revision 32's explanation is RETRACTED for numbers up to three digits.** It said: *"adding two three-digit numbers needs exact digit-position alignment and carry propagation; attention can address the tens digit exactly, a fixed state cannot."* **There are no digit positions in the prompt to align** — `437 + 562` is two tokens. Whatever arithmetic the teacher does on them happens on number-token *embeddings*, inside the network. And the carry split confirms it independently (below).

#### The scorecard

| | prediction | result | |
|---|---|---|---|
| **P0** | teacher copies | 100 % | ✅ |
| **P1** | students read fine (copy ≥ 95 %) | **100 % for all five** | ✅ **the failure is COMPUTATION, not reading** |
| **P2** | student/teacher falls with digit length | 1.00 → 0.77 → 0.10 → 0 (R1) | ✅ — but it is a sweep over **vocabulary size**, not positions, for d ≤ 3 |
| **P3** | carry hurts students more | excess over teacher **0–7 points**, below 10 | ❌ **carry propagation is not the failure** |
| **P4** | KDA > GDN | **+13.6 [+11.9, +15.3]** hybrid, **+10.0 [+8.2, +11.7]** all-linear | ✅ |
| **P5** | R5 − R1 **NULL** | **+7.6 [+6.1, +9.2]**, gap closure 0.14 → **SMALL GAIN** | ❌ **my prediction was wrong** |
| **P6** | R5 curve flat | **0.8 % → 46 %**, two collapse points | ❌ — dominated by the divergence |
| identity | `r5_ck34940` ≡ R5 top level | 2286/2286 | ✅ |

#### ⭐⭐⭐ What the students lose tracks the SIZE OF THE PROBLEM SPACE

Student accuracy as a fraction of the teacher's, ordered by the number of distinct operand pairs:

| | pairs | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|
| add_1 / mul_1 | 81 | 1.00 | ~0.97 | 1.00 | ~0.98 | 1.00 |
| mul_2x1 | 810 | 0.84 | 0.59 | 0.79 | 0.69 | 0.86 |
| sub_2 | 4,095 | 0.55 | 0.20 | 0.65 | 0.32 | 0.90 |
| add_2 | 8,100 | 0.77 | 0.36 | 0.75 | 0.42 | 0.84 |
| mul_2 | 8,100 | 0.26 | 0.17 | 0.26 | 0.21 | 0.34 |
| sub_3 | 405,450 | 0.03 | 0.01 | 0.08 | 0.03 | 0.15 |
| add_3 | 810,000 | 0.10 | 0.01 | 0.06 | 0.04 | 0.18 |
| add_4 / sub_4 | ~10⁷–10⁸ | 0 | 0 | 0 | 0 | 0 |

⭐⭐ **Small tables survive intact; the cliff is between ~10⁴ and ~10⁵ pairs.** The teacher does 3-digit addition at 94 % — 810,000 problems it cannot have memorised, so it has a **general procedure**. The students keep what a lookup could hold and lose what needs the procedure. That is a cleaner and more defensible statement than anything about digit positions: **the conversion preserved memorised arithmetic facts and destroyed the general algorithm.** 📌 *Offered as the best reading of the pattern, not a demonstrated mechanism; `mul_2` sitting below `add_2` at equal size says the operation matters too.*

⭐ **Errors are structured, not random.** Among wrong answers the units digit is right **40 %** of the time for R1 (26 % R2, 39 % R3, 31 % R4), against ~10 % by chance and **75 %** for the teacher; R5 reaches **62 %**. Operand copies are ~1 % — the "returns the first operand" pattern from two GSM8K generations does not appear here.

#### Mechanism and layout

**KDA beats GDN by 10–14 points pooled, with tight paired CIs, in both layouts** — 2× to 3× on the 2-digit conditions that have dynamic range (`add_2` 77 vs 36.5, `sub_2` 55 vs 19.5). **Layout has no effect on arithmetic**: R1 38.3 vs R3 38.7, R2 24.7 vs R4 28.7. Both replicate v1's finding that arithmetic tracks the mechanism and not the layout — on a different probe, with a paired design.

#### ⭐ R5: a real gain, smaller than the gap, from a messy run

**+7.6 points pooled, CI [+6.1, +9.2]** — concentrated on `sub_2` (**+35**), `sub_3` (+11.5), `add_2` and `add_3` (+7 each); nothing on 4-digit. Gap closure **0.14**. ⚠️ **I predicted NULL; the result is SMALL GAIN.** More FineWeb-Edu tokens *did* rebuild some arithmetic. The corpus-binding hypothesis survives only in a weakened form: the gain is real but recovers a seventh of the gap.

⭐ **The curve after the collapse rises monotonically** — 13.7 → 28.7 → 35.1 → 40.8 → 44.4 → 46.0 → 45.9 — and plateaus as the LR reaches its floor. ⚠️ **Tokens, annealing and recovery from the collapse are all confounded in that rise.** What it does show: the recovered model climbed from 14 % to 46 % in the last ~1.4 B tokens, past R1's 38 %.

### ⚠️ The batch-vs-single check fired — on the teacher too

1–2 of 8 continuations differed between batch 64 and batch 1 for the teacher, R1, R5 and several checkpoints. **The teacher shows it, so it is not a KDA defect** — almost certainly bf16 arithmetic differing with batch shape and flipping a greedy token. ⚠️ **The check compares whole 12-token continuations, most of which lie *after* the answer**, so it overstates the effect on scores. ✅ *(ran — rev 36: ~4.5 % of answers move, R5 − R1 +7.6 → +7.4)* `SET=b1` re-runs R1 and R5 at batch size 1, and `arith_agree.py` measures how many **answers** moved and whether R5 − R1 survives. The +7.6-point gap and the 10–14-point mechanism gap are far larger than anything 1–2 flipped continuations per 8 could plausibly produce, but they should be measured, not argued.

### What this does to stage 2

| finding | consequence |
|---|---|
| students read fine; they lose the **general procedure** | stage 2 is a **capability** problem, not a retrieval problem |
| more FineWeb-Edu tokens recovered **a seventh** of the gap | the corpus is not fully binding, but more of it is an expensive, slow lever |
| **1e-3 diverges** in a long run | any stage 2 or R5′: **peak LR 3e-4 or lower**, and watch `grad_norm` *(superseded rev 39–42: stage 2 uses constant 1e-5, plus a cosine 5e-5 → 1e-5 check on R1)* |
| unfreezing the MLPs | still the standard final stage. With the MLPs — where a lookup-style procedure most plausibly lives — frozen throughout, the student has had no way to rebuild the general procedure *inside* them |

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

✅ *Applied: R5 kept 10 checkpoints; stage 2 keeps 10.*

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
| **Effective tokens after packing** — ⭐ **MEASURED 2026-10-02**, 784.1/doc | **2.2903 B** = one epoch = **1,118,292** packed sequences = **34,946.6 steps** |

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

- **⚠️ The logged `grad_norm` is measured BEFORE clipping.** It can be 30× above `max_grad_norm` on a perfectly healthy step; what clipping limits is the gradient that is *applied*. In stage 2 the typical norm (~1.4) is above the 0.5 threshold, so clipping acts on every step. Judge a spike by what follows it (eval loss, later norms), not by its size alone. *(rev 44)*
- **⚠️ A summary statistic computed over the whole run counts warm-up as anomaly.** "grad_norm > 5× median" flagged 14–18 "spikes" per stage-1 run — all of them the large, normal gradients of the first ~100 steps. Compare against a running baseline after warm-up. *(rev 44)*
- **⚠️ Label eval points from the logged epoch, not from an assumed eval interval.** Stage 1 evaluates every 500 steps plus once at the end (4580); assuming max_steps/10 shifted every token label. *(rev 44)*
- **⚠️⚠️ The same metric name can be a different quantity in a different stage.** DistillKit's `eval_loss` is the configured loss on the held-out docs: in stage 1 the composite 0.2 CE + 0.2 KL + 0.6 hs_cos, in stage 2 KL alone. `summarize_run.py` v2 subtracted one from the other and printed "final vs R1 −0.2955". **Before comparing two `eval_loss` values, check that the two YAMLs have the same `loss_functions`.** *(rev 43)*
- **⚠️ A summary script that silently drops a section is worse than one that crashes.** v2 hard-coded the stage-1 loss keys; on a KL-only log the component section simply did not appear, and nothing said why. Detect what the log contains and say what was not found. *(rev 43)*
- **⚠️ A 1-GPU job with `--cpus-per-task=32` is billed as a whole node.** R = max(GPU fraction, CPU fraction) = 32/32 = 1.0 → 32 local hours per wall hour instead of 8. Every eval sbatch so far did this. *(rev 43; startup doc Part 4)*
- **⚠️ Cluster copies of scripts drift from the project copies.** The `summarize_run.py` run on 10-15 lacked the spike checks added on 10-07. `sha1sum` the cluster copy against the project before trusting its output. *(rev 43)*
- **⚠️⚠️⚠️ "Recovered" is not "undamaged".** R5's loss returned to R1's level, yet 6 % of web sequences and 14 % of Tulu 3 sequences trigger a 1,408-magnitude activation and lose 4–10 nats. **Look at the per-sequence distribution (median, worst, paired SD), not just the mean, before trusting any model that ever spiked.**
- **⚠️⚠️⚠️ A health table can be all green while the run diverges.** R5 passed every check — no NaN, no traceback, tokens agree — with `eval_loss` spiking from 0.64 to 1.75 twice and `grad_norm` reaching 83,070. **Check that the held-out curve is monotone and that `grad_norm` never explodes; `summarize_run.py` now does.**
- **⚠️⚠️ A learning rate that is safe for 4,580 steps is not safe for 34,940.** The cosine stretches with `max_steps`, so a longer run spends many times more steps near the peak. R5 held LR > 5e-4 for 17,580 steps against R1's 2,304, and diverged twice in that window and never after.
- **⭐⭐ A behavioural probe on checkpoints catches what a loss curve only hints at.** At steps 6988 and 10482 R5 could not copy a three-digit number. No loss value says that as plainly.
- **⚠️⚠️ Check the tokenizer before explaining arithmetic.** Revision 32 built a mechanism on digit alignment; AFM's tokenizer makes every 1–3-digit number a single token. **There were no digits to align.**
- **⚠️ When two measurements of the same model disagree, the larger spread names the culprit.** R5's paired SE was 20× R1's: a few catastrophic sequences, not a uniformly worse model. Look at the distribution before quoting the mean.
- **⭐ Make the batching check compare the thing you score.** Comparing whole continuations flagged mismatches on the teacher that may never touch an answer. Measure answer agreement directly.
- **⚠️⚠️ A longer run is not "the same run with more tokens".** A cosine schedule stretches to fit `max_steps`, so R5 differs from R1 in token count *and* LR history, and R5's early checkpoints are un-annealed states, not shorter runs. **Comparing across schedule lengths conflates two variables; say which one you are attributing the effect to, and that you cannot.**
- **⭐⭐ Rule out the improvement before you claim the degradation.** R5's 0-shot drop clears no single σ, but the intervals cap any gain at ~2 points — so "more tokens did not help" is solid while "more tokens hurt" is not. **The two halves of a null result have different strength; report the strong half.**
- **⚠️ Pre-registering after seeing a correlated result is still worth doing — if you say so.** The R5 predictions were written after its 0-shot numbers. Recording the timing honestly costs nothing; omitting it makes every later confirmation look stronger than it is.
- **⚠️⚠️⚠️ A distributed barrier has a 10-minute fuse, and single-process preprocessing burns it.** R5 died because rank 0 needed **606 s** to tokenize one epoch while fifteen ranks waited in `barrier()` with a **600 s** NCCL watchdog. **Any preprocessing step that scales with corpus size is a timeout waiting to happen**, and it will fire the first time you enlarge the corpus — not during development. **Warm the cache in a single-process job, where there is no barrier to expire.**
- **⚠️⚠️ The process doing the work is slowed by the processes waiting for it.** Rank 0 tokenized at **4,819 ex/s** sharing a node with three ranks spinning in a barrier, and at **6,150 ex/s** alone — 28 % faster. **A barrier is not free for the process it is waiting on.**
- **⚠️ `num_proc` is part of a `datasets` cache key.** `cache-<hash>_00000_of_00016.arrow`: change `dataset_num_proc` from 16 to 32 and the cache you just spent eight minutes building is a miss. **Raising parallelism and warming a cache are alternatives, not complements.**
- **⚠️ A `map` cache for a `load_from_disk` dataset is NOT in `HF_DATASETS_CACHE`.** It is written beside the dataset's arrow files. `$HF_HOME/datasets` being empty is not evidence that nothing was cached.
- **⚠️⚠️ A failed job can leave enough behind to block its own resubmission.** `do_distill` creates `output_path` on its first line; a trailing stamp block in a script without `set -e` then runs *after* srun fails. The `[ -e "$OUTDIR" ]` guard — which is correct and must stay — then refuses every retry. **Capture `$?` immediately after the launch and exit before stamping.**
- **⭐⭐ `bash -n` before every `sbatch`.** It costs nothing and catches the whole class of errors that a pasted, re-indented or collapsed block introduces.
- **⚠️ An interrupted `map` leaves partial cache shards under a fingerprint nothing will ever ask for.** 14 GB of `cache-73d1e91c…` at exactly 54 % of final size — the percentage where the watchdog fired. Harmless, invisible, and it accumulates.
- **⭐⭐⭐ Derive a schedule from a measurement, not an estimate — and check the divisors.** `num_train_epochs: 1` hands the step count to a runtime computation that depends on sampler padding and world size; one extra checkpoint then makes `save_total_limit` rotate away the *earliest* one, which is the point on the curve you most need. **Measure the epoch once, set `max_steps` explicitly, and make `save_steps` and `eval_steps` divide it exactly.**
- **⚠️ Read how a flag is consumed before building on what its name implies.** `functionary_packing` reads as "enable packing"; it means "assume the data is already packed in functionary's format".
- **⚠️ Confirm a token budget two ways.** R1: 146,560 configured, 146,550 measured. R2: 146,544. R3: 146,623. R4: 146,511.
- **⚠️⚠️ The last logged `distillation_loss/*` is ONE microbatch, not an endpoint.** R2's last-logged CE read 2.119 against a last-twenty mean of 2.283 — a 0.16-nat swing. Comparing single last-logged values across runs would have reversed the R3/R4 conclusion. **Cross-run comparisons use `eval_loss`.**
- **⚠️ A pre-flight constant can be layout-specific.** "Step-0 CE should be ≈ 7.94" is the *hybrid* value; the all-linear students start at **8.53–8.59**, because nine more layers are randomly initialised. A check that fires on a correct run is worse than no check.
- **⚠️⚠️ Two failures sharing one property is not a pattern when you only have five observations.** MMLU and GSM8K were the only 5-shot tasks *and* the only failures, which looked like a mechanism and was a coincidence. The 5-shot control showed the student **gains** from exemplars. **Count how many ways five things can accidentally split 3–2 before naming the split.**
- **⭐⭐ Pre-register the falsifier, then honour it.** Revision 20 wrote down "if ARC-C collapses at 5-shot it is the format; if it holds, the collapse is task-specific." It held. The prediction was cheap to write and it retired a headline in one job.
- **⚠️ A student-only benchmark number has no relative score.** The 5-shot figures cannot be turned into retention without a matching teacher run — so either run the teacher or, as here, report only the within-student comparison.
- **⚠️⚠️ A falsified hypothesis generates tidy follow-up work that tests nothing.** Killing "the shot count is the dividing line" immediately produced two proposed diagnostic jobs — neither of which touched the corpus hypothesis this document actually holds. **Ask what a job would change your mind about before queueing it.**
- **⚠️⚠️ When two rows of a table match exactly, suspect a copy before you celebrate a reproduction.** R1's start column read `7.92 / 10.13 / 0.841`, identical to R0's, for ten days. Two runs on different data samples cannot agree to three significant figures. R1's real first microbatch is 7.986 / 9.286 / 0.844.
- **⚠️ A table column needs ONE extraction method, stated.** R1's endpoint came from an epoch bin, R2's from a mean of the last twenty — both correct, neither comparable. The bin ended at epoch 0.70 while the run went to 0.7655 with KL still falling, so R1's KL read 0.35 where its endpoint is **0.314**.
- **⚠️⚠️ Two draws from the same held-out split are not the same measurement.** The layout effect changed **sign** between this project's own two held-out samples, for both mechanisms. A 0.01 effect on a quantity near 0.6 needs a paired design or ten times the magnitude before the sign means anything.
- **⚠️⚠️ Check whether your explanation survives its own arithmetic.** "A modest perplexity penalty compounds multiplicatively" sounds right and requires per-step accuracy to fall from **86 % to 40 %** — which a 17 % perplexity penalty does not do. **An explanation that cannot be made to produce the observed number is not an explanation.**
- **⚠️⚠️ Two flags bundled into one shell variable can never be used apart.** `eval_generative_afm.sbatch` had `[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT --log_samples"`, so **samples were written only on smoke runs**. Every full GSM8K and IFEval run produced an aggregate score and **no generations** — permanently. The bundling saved 13 MB and cost the most informative dataset in the project. **Log samples by default and opt out explicitly.**
- **⚠️ `--limit N` is a bad measurement of a RARE event and a fine measurement of a COMMON one.** 0/20 exact-matches is consistent with a true rate up to 14 %; a 60 % loop rate over 200 items is ±3.5 %. **The warning attaches to the quantity, not to the flag.**
- **⚠️⚠️ A decomposition can be exactly right and mean nothing.** `P(correct) = P(terminates) · P(correct | terminates)` always holds. It describes a *mechanism* only when termination is not itself caused by the thing you are explaining — and here it is. **Conditioning on a post-treatment variable produces a valid identity and a false story.**
- **⭐⭐⭐ Measure the capability directly before theorising about the task.** Three revisions of explanation for GSM8K were displaced by one 20-minute script that asked the model to add two numbers. **76 % → 20 %**, and `0.20^2.28 = 2.56 %` against a measured 2.43 %.
- **⚠️ A right mechanism can have a wrong proxy.** "The all-linear students lose the format" was correct; "loop rate" was the wrong way to measure it, and the loop ordering came out non-monotone while the termination ordering was exact.
- **⭐⭐⭐ Read the generations. A score cannot tell you a model failed to STOP.** Six of ten GSM8K outputs collapse into verbatim repetition and never emit `####`; the doc had spent three revisions attributing the collapse to arithmetic. **The failure mode was invisible in every number this project had collected** — except one, the `flexible/strict` ratio, which nobody could interpret until the text was read.
- **⭐⭐ An explanation that reproduces the observed number is worth more than one that sounds right.** `0.60⁴ × 0.40 = 0.052` against a measured 0.024 — from two independently observed rates, neither of which works alone.
- **⚠️ A correct benchmark score can come from an incorrect generation.** Doc 19 scores `exact_match 1.0` while looping twelve times; `flexible-extract` took the trailing number and it happened to be right.
- **⚠️ Three sampled generations are not an error analysis.** ✅ *(fixed rev 32: the probe ran.)* Every claim in this document about *why* GSM8K fails rests on three outputs from a 20-item smoke run, and the single-step arithmetic probe that would settle it has never been run.
- **⭐⭐⭐ Two metrics that disagree may be measuring different regimes, not contradicting each other.** Held-out CE/KL and three 0-shot benchmarks favoured the all-linear layout; MMLU and GSM8K, the only long-prompt tasks, favour the hybrid at over 4 σ. **Before reconciling a disagreement, check whether the measurements share an operating point.**
- **⭐⭐ A scoring filter can be an independent instrument.** GSM8K's `flexible/strict` ratio — 1.44×, 1.73×, 3.93×, **40.75×** — measures whether the model reproduces a format shown five times in the prompt. It corroborated MMLU's layout result from a completely different task, for free, out of numbers already printed.
- **⚠️⚠️ A YAML's `output_path` is the only thing that decides where a run writes.** On 2026-09-29 a resume test built by editing selected fields kept R1's `output_path` and **overwrote R1's trained model with a 20-step one**. `checkpoint-4580` survived because `save_total_limit` rotates by step number. **Never hand-edit a subset of fields in a copied config; write the whole file, and guard `output_path` in the sbatch.**
- **⭐⭐ Decompose a composite metric before explaining it.** The composite is 60 % `hs_cosine`, which made "the disagreement is representational-vs-behavioural" irresistible — and wrong. `hs_cosine` contributes **−12 %** of the layout gap. **A weight in the objective is not a share of the effect.**
- **⭐ Look at the component with the widest spread.** CE varies 1.8 % across four architectures and KL varies **20.6 %**. Averaging them into one number threw away the discriminating signal.
- **⚠️⚠️ A training objective is not a quality metric, and it ranks some axes wrongly.** Held-out composite loss is 60 % `hs_cosine`, which measures *representational* similarity to the teacher. It ranks the two mechanisms the same way the benchmarks do — and ranks the two **layouts the opposite way**, because an all-linear student is structurally further from the teacher's internals while behaving just as well. **Rank layouts on benchmarks, not on `eval_loss`.**
- **⚠️ A pooled p-value built from within-task standard errors is not a claim about your experiment.** Z ≈ 3.1 across six benchmark comparisons says the *direction* is unlikely to be an accident of which questions are on the test. It says nothing about training-seed variance, which no run here has measured.
- **⚠️ `grep -c` counts LINES; per-rank prints share lines.** Sixteen ranks writing one stdout put `Froze 147 tensors` twice on three lines, so `-c` read 13 where the truth was 16. **Use `grep -o PATTERN F | wc -l` for anything printed once per rank.**
- **⚠️ An unanchored health-check pattern hits your own log header.** `grep -i nan` matches "prove**nan**ce". Use `grep -iE "\\bnan\\b"`.
- **⭐ Replicate across a second axis when you cannot afford a second seed.** Repeating the mechanism comparison at a different layout is not as good as two seeds, but it is far better than one run — and it produced the per-layer scaling that a seed repeat would not have.
- **⚠️⚠️ `grep -c` counts matching LINES, not matches.** R2's log showed `Froze 147 tensors` on 13 lines and the check read as a failure against the expected 16 — but three lines carried the string **twice**, because sixteen ranks write to one stdout and their writes interleave. `grep -o ... | wc -l` returned 16. **Count occurrences, not lines, for anything printed per rank.**
- **⚠️ `grep -i nan` matches "provenance".** A health check that greps for `nan|error|traceback` over a whole log will hit the word in your own header. Anchor the pattern or check the follow-up before believing a hit.
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