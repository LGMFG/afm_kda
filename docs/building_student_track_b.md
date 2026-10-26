# Track B — Generation ✅ **COMPLETE — the student writes text. B8 and B9b are now ordinary evaluation jobs.**

*Companion to `claude/building_the_student.md` (Track A — how the student was built) and `claude/first_distillkit_runs.md` (the live training document). This one covers **generation**: giving the student a working cache so it can write text, not just score it.*

*Revision 16 (2026-09-25) — ✅✅ **§4's last open caveat is retired.** B6e re-run at **128 decode steps** on `afm_kda_only_init`: the fp32 per-position error goes **7.312e-04 → 2.087e-03**, so the *final* position is **four times cleaner than at 32 steps**, the worst case rises only 10 % for 4× the length, the bf16 median is **unchanged**, and argmax is **128/128**. The curve is **flat, not merely bounded** — long generations carry no drift risk, and a linear extrapolation written down beforehand was wrong. ✅ **B7c also re-run on the re-synced `output/r1_stage1_hybrid31`**: fluent prose on all three rows, `use_cache: True`, and the most-padded row again the best — which is the C-5 stash loop finally exercised through `generate()` on real KDA layers.*

*Revision 15 (2026-09-25) — ✅ **B8's GSM8K full run landed**: teacher **0.5353**, student **0.0243** strict on all 1,319 items — **not zero** (5.8 σ, ~32 problems), and the student loses the `####` format as well as the arithmetic. ⚠️⚠️ **§7's account of *why* the benchmark spread looks the way it does is falsified by MMLU** (13.9 % relative, on a likelihood-scored task): the dividing line is the **shot count**, not likelihood-against-generation. ⭐ Also adds the **untrained B7 baseline** to §4 — 6–13 ulp and 27–30/32 across three students, against a trained model's 1–2.5 and 31–32/32, with a zero-padding control at 6.8 ulp that puts most of the magnitude outside padding entirely.*

*Revision 14 (2026-09-25) — ⭐ **B6e re-run on `afm_kda_only_init`**, the all-linear student: fp32 worst 1.439e-02 against the hybrid's 8.021e-03, 32/32 argmax, and **growth with position unchanged at 8.1× where the hybrid reads 9.1×** — so stripping out all nine attention anchors did *not* make state drift worse. §4 has the table. ⚠️ Two corrections come with it: the script's own printed verdict threshold (1e-5) **predates the TF32 finding and would fail the validated hybrid**, and the run proves **less** about the `attn_layer_idx is None` fallbacks than it appears — in an all-linear model nothing consumes their answer, so they are *inert*, not *validated*. **B7 on an `only` student is the job that actually gates one.**

*Revision 13 (2026-09-21) — **restructured.** Glossary removed; the terse status board replaced by a "what we set out to do / what we actually did" table; the applied patch and the full script sources removed in favour of pointers, since both now live in git (`afm_kda` @ `d7cd0a0`) and in the project folder. Content is unchanged except where marked.*

*Revision 12 (2026-09-21) — ⭐⭐ **B7 CLOSED.** Two independent defects, both confirmed by reading transformers 5.8.1 and fla 0.5.2 and then measured: the causal mask was being built one column wide, and the 2-D padding mask never reached the KDA layers. Both fixed; B7 passes under sdpa and flash-attention.*

*Revisions 1–11 (2026-09-19 → 09-21) — B0 probe and the abandoned fla-Cache-everywhere design; B9a memory curve; the B2′–B4′ two-cache bridge; B5/B6/B6b/B6c/B6d/B6e; three invented tolerances and the control that replaced them; B7 attempts 1 and 2. All of it is folded into §2 and §3 below; the blow-by-blow is in the conversation history, the conclusions are here.*

---

## 1. Why this track exists

The student can **score** text — hand it a passage and it returns the probability of every token. It could not **write** text, and the way it failed was silent.

Writing means producing one token at a time, and no model recomputes the whole prefix at every step; it keeps a **cache**. The nine attention layers keep a KV cache that grows with the text. The twenty-seven KDA layers should keep a **recurrent state** — a fixed 20 × 128 × 128 matrix, the same size at token 30,000 as at token 30. Our adapter originally hardcoded `past_key_values=None, use_cache=False`, so they kept nothing. Continue from a cache and the attention layers would be right while the KDA layers behaved as if the text had just started: fluent, wrong, no error anywhere.

| | needs generation? | status |
|---|---|---|
| MMLU, ARC-C, HellaSwag, WinoGrande | ❌ one forward pass, likelihood-scored | ✅ unaffected throughout |
| GSM8K | ✅ | ✅ unblocked |
| IFEval — 2048 new tokens per prompt | ✅ | ✅ unblocked |
| **Decode throughput vs context length** (B9b) | ✅ | ⬜ the headline number still to measure |
| Inference **memory** vs context length (B9a) | ❌ a cache is built by any `use_cache=True` forward | ✅ measured |

⚠️ **A capability you cannot exercise is a claim you cannot make.** KDA's smaller inference state is the point of the project, and half of it — the speed half — was unmeasurable until this track landed.

---

## 2. What we set out to do, and what we actually did

| Step | What we set out to do | What we actually did | Artifact | How we knew |
|---|---|---|---|---|
| **B0** | Find out whether fla's own `Cache` can simply replace transformers' everywhere, the way the reference does it | A ten-minute probe: load the student, call `forward` with an fla `Cache` | `test_b0_probe.py` | ❌ **It raises inside transformers' own model wrapper, before any of our code.** Also measured `get_seq_length() -> 0` while `__len__() -> 36` — the first sighting of the layer-0 lie that ran through the whole track |
| **B0b** | Read the thirty lines of transformers that raise, before committing to a redesign | Read them | — | **Inconclusive, and deliberately dropped** — B2′ makes the question unreachable |
| **B9a** | Measure inference **memory** against context length, for teacher / hybrid / kda_only | Forward with `use_cache=True`, walk the returned cache, sum every tensor, cross-check against closed form | `test_b9_cache_memory.py` | ✅ At 32k: teacher **2,304 MB**, hybrid **611 MB (3.77×)**, kda_only **47.1 MB (48.9×)**. ⭐ And a crossover at **T ≈ 670**, below which a converted model carries *more* state than the teacher |
| **B2′–B4′** | Build a bridge that leaves `ArceeModel.forward` untouched — the thing A3 depends on | `AFMKDACache(DynamicCache)` carrying an fla `Cache` as an attribute; installed in `AFMKDAModel.forward`; the KDA layers take it off the carrier | 5 edits to `modeling_afm_kda.py` | verified by B5 |
| **B5** | Prove the patch did not reach the **training** path | Re-ran A3, the Track A regression test, twice | `test_a3.py` | ✅ **0.000e+00** on all five checks |
| **B6** | Does the cached path produce the same text as no cache at all? | Three paths on a real 300-token prompt: full-recompute oracle, our cache by hand, and `generate()` | `test_b6_cache_equiv.py` | ✅ **A == B == C, 64 tokens each, token for token.** And `get_seq_length() -> 300` where B0 measured 0 |
| **B6b** | Rule out "they agree because *neither* is caching" | Walked `pkv.fla` after the prompt forward and asserted it is non-empty | same script | ✅ **35.33 MB stored.** ⭐ And the number disagreeing with my predicted 18.06 was itself the finding: the recurrent state is **fp32**, and the conv state is `conv_size` wide |
| **B6c** | Does it hold on other prompts, for longer, with the state staying constant? | Three prompts × 128 free-running greedy tokens | same script | ⚠️ **1 of 3 diverged — and the test was at fault.** Greedy decoding is a chaotic map; both divergences were genuine near-ties (top-2 gaps of half an ulp and one ulp). State constant at 35.33 MB throughout |
| **B6d** | Ask the same question without chaos | **Teacher-forced** comparison — drive both paths along identical tokens | `test_b6d_precision.py` | 2–3 ulp, flat with position. ⭐ And reading `kda.py` explained why bit-identity was never available (see §4) |
| **B6e** | Separate rounding from a bug **without inventing a tolerance** | Same comparison in **fp32**, plus the unmodified teacher through the identical protocol as a control | same script | ✅ **The teacher disagrees with itself by exactly the same 1.875e-01.** fp32 shrinks the student's gap 23× to 8.0e-03 with 32/32 argmax. **The bridge is correct** |
| **B7** | Does it survive the way lm-eval actually generates — several prompts of different lengths, **left-padded**, in one batch? | Four rows (120/180/240/300) padded to 300, 32 teacher-forced tokens, teacher as control | `test_b7_batched.py` | ⚠️ Crashed twice before it could answer (§3), then ✅ **PASSED** under sdpa and flash-attention |
| **B7b** | Stop guessing: **print** what each per-layer accessor answers, and the mask's stride | Probe with a forward pre-hook on the first attention layer | `test_b7b_mask_probe.py` | ✅ Student layers 0/1/2 answer `0`, layer 3 answers `300`. After the fix the decode mask is `(4,1,1,301)`, contiguous, **identical to the teacher's**, and the step survives |
| **B7c** | Close the last gap: every test so far hand-rolled the decode loop; lm-eval calls `generate()` | Three prompts, left-padded, 48 greedy tokens, teacher first | `test_b7c_generate.py` | ✅ **PASSED 2026-09-22.** Fluent, factually correct prose on all three rows. ⭐ The row with the **most** padding (11 of 17 columns) gave the best output. `config.use_cache` printed **True** as loaded. ⚠️ And it caught the teacher loading with `use_cache=False` — see §6 |
| **B8** | The generative benchmarks | GSM8K 5-shot + IFEval 0-shot, teacher first, `use_cache=True` forced on both | `eval_generative_afm.sbatch` | ✅ **GSM8K FULL, 1,319 items:** teacher **0.5353**, student **0.0243** strict. ⭐ **Not a Track B failure:** the same prompts with `use_cache=False` give **byte-identical** output. ⬜ IFEval still to run. See §7 |
| **B9b** | Decode **throughput** vs context length — the half of B9 that needs the bridge | ⬜ | ⬜ | ⬜ |

**The original B2/B3/B4 — convert everything to fla's `Cache`, as Arcee do — are dead.** B0 killed them in one debug job. The primed numbers are the replacement, and there is no separate "regular" track to keep straight.

---

## 3. ⭐⭐ The bug, end to end

Two independent defects in two different halves of the model. Both were established by **reading** transformers 5.8.1 and fla 0.5.2 on the login node during a scheduler outage, then confirmed by measurement — which is the opposite of how the first two-thirds of this track went.

### Defect 1 — the causal mask was built one column wide

`DynamicCache.get_mask_sizes` in 5.8.1 *already* handles hybrids, but only when the linear layers are **registered** as non-`CacheLayerMixin` entries:

```
LAYER_TYPE_CACHE_MAPPING:
  full_attention    -> DynamicLayer             CacheLayerMixin=True
  linear_attention  -> LinearAttentionLayer     CacheLayerMixin=False   <- the marker
  mamba / conv      -> LinearAttentionLayer     CacheLayerMixin=False
```

Ours are registered as **nothing**. `AFMKDACache()` is constructed bare, so `DynamicCache.__init__` falls through to `layer_class_to_replicate=DynamicLayer` and lazily back-fills indices 0–2 with **empty** `DynamicLayer`s the moment layer 3 first writes. Index 0 then looks like a perfectly ordinary attention layer holding nothing, the rescue branch never fires, and:

```python
DynamicLayer.get_mask_sizes:  kv_length = self.get_seq_length() + query_length
                                        = 0 + 1 = 1          # a decode step
```

A mask of `[B,1,1,1]` against 301 keys is *broadcastable*, so **no shape error is raised**. PyTorch broadcasts it; a broadcast dimension has **stride 0**; SDPA's kernel check `bias.stride(-1) == 1` fails with `(*bias): last dimension must be contiguous`.

⭐ **That is why `.contiguous()` could not help** — the offending stride is created *inside* SDPA, after our call. And it is why **B6 passed while B7 failed**: B6 passes no `attention_mask` at all, so transformers takes the no-mask path with `is_causal=True` and never calls `get_mask_sizes`. The bug is only reachable once padding forces a real mask to be built.

```python
    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        if self.attn_layer_idx is None:
            return query_length, 0
        return super().get_mask_sizes(query_length, self.attn_layer_idx)
```

*(Ignoring the incoming `layer_idx` is correct here because this model has one attention type; it would **not** be correct for a sliding-window hybrid.)*

### Defect 2 — the padding mask never reached the 27 KDA layers

`AFMKDAModel.forward` calls `super().forward()` → `ArceeModel.forward` → `create_causal_mask`, so the mask arrives 4-D. fla refuses anything else:

```python
assert len(attention_mask.shape) == 2, "Expected attention_mask as a 0-1 matrix [batch, seq_len] ..."
```

so the adapter's `dim() != 2 -> None` line prevented a crash — and thereby let pad tokens into the recurrent state. What it was throwing away is fla's own handling:

```python
indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
hidden_states = index_first_axis(...)                     # strip pads, run varlen
o = pad_input(o.squeeze(0), indices, batch_size, q_len)   # put them back
```

That path is side-agnostic, so left padding is fine — **if the mask arrives**. The fix stashes the raw 2-D mask on each KDA module before `super().forward()` mangles it, and the adapter prefers it over whatever arrived.

### The shape of it

```
              ONE DECODE STEP    (300 tokens seen, 1 new token arriving)

  generate() -> AFMKDAModel.forward
                     |
                     |  (1) "how many tokens are behind us?"   cache.get_seq_length()
                     |  (2) "how wide must the mask be?"       cache.get_mask_sizes(1, 0)
                     v
        +-------------------- AFMKDACache --------------------+
        |   DynamicCache half            fla Cache half       |
        |   (self.layers)                (self.fla)           |
        |   0   (empty)   <- lazily      0   state  1.3 MB    |
        |   1   (empty)      created     1   state  1.3 MB    |
        |   2   (empty)      by 3's      2   state  1.3 MB    |
        |   3   300 keys *   write       3   --               |
        |   4   (empty)                  4   state  1.3 MB    |
        |   ...                          ...                  |
        |   35  (empty)                  35  state  1.3 MB    |
        +-----------------------------------------------------+
             9 real entries                27 real entries
             grow with the text            fixed size forever

  Both (1) and (2) are asked of THE CACHE, once, and default to layer 0.
  Layer 0 is KDA.  Both overrides re-route them to * (index 3).

  attention_mask  [4, 301]   (0 = padding)
        |
        +- create_causal_mask -> [4,1,1,301] 4-D bias -> 9 ATTENTION layers
        |                        width comes from (2)     SDPA needs 4-D
        |
        +- stashed raw 2-D ---------------------------- > 27 KDA layers
                                                          fla asserts 2-D, then
                                                          unpads -> varlen -> repads
```

⭐ **The KDA layers do have a cache** — a fixed 20 × 128 × 128 recurrent state plus a conv state, in the fla half. What they have no notion of is a *sequence length*, because their memory does not grow. `get_seq_length` and `get_mask_sizes` are questions about **KV geometry**, and KDA has no KV. That is the whole of both defects.

**Layer counts:** 36 layers, **27 KDA and 9 attention** — three KDA then one attention, repeating, so attention sits at indices 3, 7, 11 … 35. `hybrid31` is the 3:1 ratio, not a layer count.

### For a pure-KDA student (Track C)

`attn_layer_idx` is `None`, and both branches degrade correctly: `get_seq_length()` falls back to `self._counted`, maintained by `note_tokens`, because no layer anywhere could answer; `get_mask_sizes` returns `(query_length, 0)`, a value nobody consumes since a causal mask is only ever read by an attention layer. Transformers' own code would **raise** there (*"the current Cache seem to only contain LinearAttention layers"*), so the fallback is strictly safer than delegating. **Defect 2 matters far more in that model**: with all 36 layers linear, a missing padding mask makes left-padded batched generation entirely wrong rather than partly.

### What we tried before reading the source

Recorded because the pattern is the lesson, not because these are live options.

| attempt | why it seemed right | what actually happened |
|---|---|---|
| `attention_mask.contiguous()` in `AFMKDAAttention.forward` | SDPA names contiguity in the error | ❌ The bad stride is created inside SDPA, after our call. Harmless, still in the file |
| `IMPL=flash_attention_2` (Fix A) | flash never builds a 4-D mask, and it is what the reference and R1 both run under | ⚠️ **Works, but by bypassing the problem.** It also means that run said *nothing* about Defect 2, since the mask arrives 2-D by itself |
| Three invented absolute tolerances (1e-2, then 2 ulp, twice) | they looked like reasonable noise floors | ❌ All three misled the track. Replaced by a **control** (the teacher through the identical protocol) and by **changing the precision** |

---

## 4. The numbers

### B6e — is the bridge correct? *(teacher-forced, 128-token prompt + 32 forced tokens)*

| | worst \|Δlogit\| | median | argmax agreement |
|---|---|---|---|
| student, bf16 | 1.875e-01 | 1.250e-01 | 31/32 |
| **student, fp32** | **8.021e-03** | 4.251e-03 | **32/32** |
| **teacher, bf16** *(control)* | **1.875e-01** | 1.250e-01 | **31/32** |

⭐⭐ **The teacher row is the result.** The unmodified all-attention teacher, using transformers' own KV cache, disagrees with *itself* by exactly the same 1.875e-01 and gets the same 31/32. For our number to be a bug it would have to be a bug that coincidentally reproduces a correct implementation's magnitude to three digits.

**Why bit-identity was never available.** `kda_installed_0.5.2.py` line 211:

```python
mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
```

KDA picks a **different Triton kernel** by sequence length. A 428-token pass runs `chunk_kda`; the cached path runs `chunk_kda` for the 300-token prompt and then `fused_recurrent_kda` for every single-token step. Different kernels, different accumulation orders. The two paths are not the same arithmetic, so "the same computation through a cache" is literally a different program.

**Why 8e-3 and not 1e-7 in "fp32".** Triton's `tl.dot` uses **TF32** on Ampere by default — 10-bit mantissa, ε ≈ 4.9e-4 — so 3.7e-4 relative is the floor the kernels actually run at, not drift.

**Why the recurrent state is fp32 while the model is bf16.** It is a kernel *accumulator*, not a parameter: updated in place across thousands of steps, where bf16's 8 significant bits would round small updates away entirely. Attention never faces this because it stores K and V and re-sums them in a fresh accumulator each time. Every parameter in the model is still bf16 — A3 confirms 4.999 B unchanged. Mamba keeps its `ssm_state` in fp32 for the same reason.

📌 **One caveat, recorded rather than waved away.** In fp32 the per-position error goes 7.863e-04 → 7.191e-03 over 32 steps, ~9× growth, invisible in bf16 under the 1.25e-01 quantisation step. Mild state drift and "early positions agree better by luck" both fit. It does not change the decision — even growing linearly it stays under bf16's noise floor for hundreds of tokens, and the teacher's own first→last moves too (*decreasing*, 1.748e-01 → 1.328e-01). ✅ **CHECKED at 128 steps, 2026-09-25 (job 58645600), and the caveat is retired — the curve is not merely bounded, it is FLAT.**

| `afm_kda_only_init`, fp32 | 32 steps | **128 steps** |
|---|---|---|
| first → last position | 1.116e-03 → 8.989e-03 | **7.312e-04 → 2.087e-03** |
| growth across the window | 8.1× | **2.9×** |
| **last position** | 8.989e-03 | **2.087e-03** — *four times SMALLER* |
| worst \|Δlogit\| | 1.439e-02 | **1.579e-02** — +10 % for 4× the steps |
| argmax | 32/32 | **128/128** |

⭐⭐ **The "~9× growth" was an artefact of where the 32-step window happened to end, not a trend.** Quadruple the decode length and the *final* position's error drops by 4× and the worst case rises 10 %. In bf16 the same flatness is explicit: student **3.281e-01 → 3.125e-01** against the teacher's **1.250e-01 → 1.250e-01**, and the bf16 **median is unchanged at 2.500e-01** between the two runs — only the *max* grew (4.8 → 15.5 ulp), which is what a max over four times as many draws from a stationary distribution does.

⚠️ **A prediction written down before the run was wrong, and is recorded as wrong.** Extrapolating the 32-step slope linearly gave ~2.5e-4 per token and ~3.3e-02 at step 128, with the worry that IFEval's 2,048 tokens would approach one bf16 ulp. **The slope is zero, not 2.5e-4.** Long generations carry no accumulating-drift risk. *Trap: a two-point slope through a noisy window is not a trend — and "does the error grow?" is only answerable over a window long enough for growth to show.*

### ⭐ B6e on `kda_only` — the same test with **no attention layers at all** *(2026-09-25)*

`sbatch --export=ALL,MODEL=$WORK_LUKAS/students/afm_kda_only_init test_b6_cache.sbatch` — 36 linear layers, zero attention, same 128-token prompt + 32 forced tokens.

| | worst \|Δlogit\| | median | first → last | argmax |
|---|---|---|---|---|
| hybrid, bf16 | 1.875e-01 *(1.5 ulp)* | 1.250e-01 | — | 31/32 |
| **`only`, bf16** | **1.188e+00** *(4.8 ulp @ 35.0)* | 2.500e-01 | 2.266e-01 → 3.750e-01 | **31/32** |
| hybrid, fp32 | 8.021e-03 | 4.251e-03 | 7.863e-04 → 7.191e-03 | 32/32 |
| **`only`, fp32** | **1.439e-02** *(0.1 ulp)* | **1.266e-03** | 1.116e-03 → 8.989e-03 | **32/32** |
| teacher, bf16 *(control)* | 1.875e-01 | 1.250e-01 | 1.748e-01 → 1.328e-01 | 31/32 |

⚠️ **The verdict text the script prints is stale.** It asks for "~1e-5 or below", a threshold written *before* the TF32 finding two paragraphs up. **The hybrid — validated four ways since — reads 8.021e-03 and would fail that criterion too.** The live question is not "is it below 1e-5" but "is 1.439e-02 the same phenomenon as 8.021e-03". Three readings say yes:

- **Precision scaling: 82.6×** (1.188e+00 → 1.439e-02), against the hybrid's 23.4×. Rounding shrinks by orders of magnitude between bf16 and fp32; a logic error shrinks by 1.
- **⭐ Growth with position is unchanged: 8.1×** over 32 steps against the hybrid's 9.1×. **This is the decisive row.** Removing all nine attention layers removes every point in the stack where the two paths compute *the same function*, so if the fp32 drift in the caveat above were accumulating state error, 36 unanchored layers would make it **worse**. It does not.
- **Argmax 32/32 in fp32 and 31/32 in bf16 — matching the teacher control exactly.** And the `only` model's fp32 *median* is 1.266e-03, **3.4× lower than the hybrid's**. The worst case is a max over 32 × 128,004 logits and is one outlier; the median is the robust statistic.

⭐ **Why bf16 got worse while fp32 barely moved.** In the hybrid, nine of thirty-six layers are attention and run the *same* arithmetic on both paths, re-anchoring the residual stream eight times over. In `kda_only` every layer runs `chunk_kda` on one path and `fused_recurrent_kda` on the other, with nothing exact in between. 1.5 ulp → 4.8 ulp is that, and it is an arithmetic property of the kernel switch, not a defect.

📌 **The teacher control reproduced to every digit** — 1.875e-01, median 1.250e-01, 1.748e-01 → 1.328e-01, 31/32, identical to B6e's row above. The harness is deterministic and the control is a genuine constant rather than a coincidence.

#### ⚠️⚠️ But it proves **less** about the all-linear fallbacks than it looks

The run was proposed as *the* test of `AFMKDACache`'s `attn_layer_idx is None` branches. It is not, and the reason is worth writing down: **in a model with no attention layers, nothing consumes the answer.**

```
get_seq_length()  ->  cache_position  ->  position_ids  ->  RoPE
                                                             |
                                    no linear layer reads position_embeddings

get_mask_sizes()  ->  the causal mask  ->  read only by an attention layer
                                           (and the B6 family passes no mask at
                                            all, so it is never even called)
```

So `_counted` could be **permanently zero** and this test would still pass. The fallbacks are not *validated* here; they are **inert** — a different and weaker claim, though a reassuring one, because inert code cannot be wrong in a way that matters.

**What the run did establish, and it is not nothing:** the recurrent state carries correctly through thirty-six linear layers with no re-anchoring, at the same precision character as the hybrid, with token-level agreement matching the control.

**What actually gates an `only` student is Defect 2, not the fallbacks.** §3 says it already: with all 36 layers linear, a padding mask that fails to arrive makes left-padded batched generation *entirely* wrong rather than partly. That is the configuration lm-eval uses, and it is untested for either `only` variant. ⬜ **B7 on `afm_kda_only_init` is the job that matters.**

⬜ **And the 128-step check now has a better subject.** The caveat above asks whether the fp32 curve is bounded; `kda_only` is the more sensitive probe because it has no anchors. The slope matters concretely: ~2.5e-4 per token, extrapolated linearly, reaches one bf16 ulp (0.25 at scale 35) somewhere near a thousand tokens — and **IFEval generates 2048**. In bf16, the condition that actually runs, the same curve grows only 1.65× over 32 steps because quantisation dominates. One debug job bounds it.

### B7 — batched, left-padded *(4 rows, 32 forced tokens each)*

| impl | row | pad | teacher ulp / argmax | student ulp / argmax |
|---|---|---|---|---|
| **sdpa** | 0 | 180 | 1.8×  31/32 | 1.5×  **32/32** |
| | 1 | 120 | 1.5×  32/32 | 2.5×  31/32 |
| | 2 | 60 | 1.2×  32/32 | 1.1×  32/32 |
| | 3 | 0 | 1.5×  31/32 | 1.5×  31/32 |
| **fa2** | 0–3 | 180…0 | 1.2–1.9×  31–32/32 | 1.4–2.5×  30–32/32 |

⭐ **The decisive property is the ordering, not the magnitudes.** Row 0 carries the most padding and is the student's *cleanest* row; row 3 carries none and is among its worst. Pad contamination of the recurrent state would invert that and cost O(1–10), not 2 ulp.

⚠️ **Only the sdpa run tested the patch.** Under flash-attention `create_causal_mask` short-circuits and the mask never becomes 4-D, so it reaches the KDA layers 2-D by itself and the stash is never exercised.

*One unit in the last place (ulp) is the smallest difference the format can represent **at that magnitude**. bf16 keeps ~8 significant bits, so near a logit of 22 the spacing is 0.125 and near 34.5 it is 0.25. An absolute tolerance without a magnitude is meaningless — the mistake this track made three times.*

### ⭐ B7 on three more students — the untrained baseline, 2026-09-25

Run for Track C, recorded here because it is a Track B measurement and it establishes a baseline B7 never had.

| student | state | pad 180 | pad 120 | pad 60 | pad 0 | argmax |
|---|---|---|---|---|---|---|
| teacher *(control)* | — | 1.8 ulp | 1.5 | 1.2 | 1.5 | 31–32/32 |
| hybrid31 | **trained** | 1.5 | 2.5 | 1.1 | 1.5 | 31–32/32 |
| `kda_only` | untrained | 7.5 | **9.3** | 5.8 | **8.5** | 29–30/32 |
| `gdn_hybrid31` | untrained | **10.6** | 10.1 | 5.5 | 6.9 | 27–29/32 |
| `kda_hybrid31` | untrained | **8.1** ← *best row* | 12.9 | 12.0 | **12.0** | 28–29/32 |
| `gdn_hybrid31`, **one row, zero pad** | untrained | — | — | — | **6.8** | 28/32 |

⭐⭐ **Two things settle it.** The zero-padding row reads **6.8 ulp at batch 1** — so most of the magnitude exists *before any pad token does*, and is the cost of comparing a masked varlen path against an unmasked dense one on an untrained model. And the untrained **KDA** hybrid is **worse** than the GDN one, with its *most-padded* row the **cleanest** — the opposite gradient. **Pad contamination does not produce opposite orderings in two models running the same stash loop.**

⭐ **The untrained baseline is 6–13 ulp and 27–30/32**, against a trained model's 1–2.5 ulp and 31–32/32. ⚠️ **B7's published pass band was measured on a trained student and does not transfer to an untrained one** — argmax agreement is a function of how peaked the distribution is, and an untrained student sits at perplexity 2,765, where near-ties are everywhere.

⚠️ **`LENS=120,120,120,120` silently collapses to a batch of one.** The script builds its batch from *distinct* lengths, so the intended four-equal-rows test never ran. It answered the question anyway — better, in fact, since batch 1 with no padding is the cleanest possible floor — but read the `batch N | lengths [...]` header before trusting what was tested.

### B9a — inference memory *(measured; the companion is B9b, still to run)*

| context | teacher | **hybrid 3:1** | **kda_only** |
|---|---|---|---|
| 512 | 36.0 MB | 44.3 MB — **0.81×** | 47.1 MB — **0.76×** |
| 2,048 | 144.0 MB | 71.3 MB — 2.02× | 47.1 MB — 3.06× |
| 8,192 | 576.0 MB | 179.3 MB — 3.21× | 47.1 MB — 12.2× |
| 32,768 | 2,304.0 MB | 611.3 MB — **3.77×** | 47.1 MB — **48.9×** |

Per KDA layer: `20 × 128 × 128 × 4 B (fp32) + 3 × 2560 × 4 × 2 B (bf16) = 1,372,160 B = 1.3086 MB`, exact against measurement. ⭐ The crossover is at **T ≈ 670** for *both* converted models, because it is a per-layer property: each converted layer trades `2 × T × 512 × 2 B` of growing KV for 1.37 MB of fixed state.

---

## 5. The patch — seven edits, all applied and committed

`afm_kda` @ **`d7cd0a0`**, clean tree. The full source is in git and in `modeling_afm_kda.py` in the project folder; only the shape of it is worth keeping here.

| # | Where | What |
|---|---|---|
| 1 | module top | import `Cache as FlaCache` and `DynamicCache` |
| 2 | above `AFMKDAAttention` | `class AFMKDACache(DynamicCache)` — carries an fla cache as `.fla`, overrides `get_seq_length` (**B2′**) and `get_mask_sizes` (**B7**), plus `note_tokens`/`_counted` for the all-KDA case |
| 3 | `AFMKDAModel.forward` | install the cache; `_first_attn_layer_idx()`; `_is_stateful = True` (**B3′**) |
| 4 | `AFMKDAModel.forward` | stash the raw 2-D padding mask on each KDA module before `super().forward()` (**B7**) |
| 5 | `AFMKDALinearAttention.forward` | take the fla cache off the carrier; prefer the stashed 2-D mask over the 4-D one (**B4′ + B7**) |
| 6 | `AFMKDAAttention.forward` | `attention_mask.contiguous()` — ⚠️ **dead weight**, kept only because removing it costs a sync cycle |
| 7 | module bottom | import-time asserts, in the model file rather than in a test, so a stale checkpoint cannot load quietly |

⚠️ `_is_stateful` turned out to be a **guard rail, not a mechanism**: its one call site in the whole generation stack refuses *assisted generation* on stateful models. Setting it is correct and costs nothing; it fixes nothing.

---

## 6. Getting a checkpoint ready to evaluate

### Provenance — correct by construction, not by remembering

```
afm_kda/ under git  ->  students/*_init built from it  ->  save_pretrained carries
the .py files into output/<run>  ->  the commit hash identifies the code
```

`save_pretrained` copies a trust-remote-code model's `.py` files from the loaded directory into the output, which is why `output/r1_stage1_hybrid31` had a copy in the first place. So a committed `afm_kda` plus rebuilt init students makes every future run correct **by construction**. `finalize_checkpoint.py` therefore does **not** copy by default; it *verifies* and flips.

| what | how | why not inherited |
|---|---|---|
| the model code | inherited through the chain above | — |
| `use_cache` | flipped by the script | Trainer overrides it in memory at save time, whatever the init config said |
| the markers | checked by the script | the chain is **silent** when it breaks: an init student not rebuilt, a run launched from an older init |

```bash
python $WORK_LUKAS/afm_kda/finalize_checkpoint.py \
    $WORK_LUKAS/output/r1_stage1_hybrid31 \
    $WORK_LUKAS/output/r1_stage1_hybrid31/checkpoint-4580
rm -rf $HF_HOME/modules/transformers_modules
```

✅ Both paths **PASS**, 2026-09-21: 5/5 markers, `auto_map` ok, `use_cache False -> True` in both JSON files. `--sync` (copy) is opt-in and exists only to retrofit a pre-fix checkpoint; **needing it on a fresh run is itself a finding.** Add one marker line to that script every time a fix lands.

### Why `use_cache` needs flipping, and why that is not cheating

HF Trainer sets `config.use_cache = False` whenever gradient checkpointing is on, and `save_pretrained` serialises it. Left alone, generation still produces the *right* text — it just recomputes the entire prefix at every token, O(n²) instead of O(n). It changes **nothing** for the likelihood-scored benchmarks; it makes GSM8K and IFEval all-nighters; and it **invalidates B9b outright**, since a decode-throughput curve without a cache measures exactly the thing the architecture is supposed to eliminate.

`use_cache` is not a learned parameter, not an architectural choice, not a hyperparameter. It selects between two ways of computing **the same function**, and B6, B6d/B6e and B7 have now proved that equivalence three separate ways. What *would* be cheating is measuring teacher and student under different settings and reporting the ratio. **The rule: the flag is identical for both, and stated once in the methods section.**

⚠️⚠️ **And the teacher is not flipped — which makes every timing comparison a trap.** B7c printed `config.use_cache: False` for `AFM-4.5B-Base` and `True` for our student, because `finalize_checkpoint.py` only ever touches ours. Left alone the teacher runs O(n²) and the student O(n), and a speed or wall-clock comparison between them would be exactly the dishonest measurement this section warns against. **Fix: pass `use_cache=True` in `model_args` for both models**, which puts them on the same footing whatever their config files say and touches no checkpoint. `eval_generative_afm.sbatch` does this.

*Trap: **a setting you fixed on one model is a setting you have now made asymmetric.** Flipping `use_cache` on the student silently created the confound; the teacher's config was never ours to forget.*

⚠️ **Arcee do not flip it either.** `arcee-ai/AFM-4.5B-Base-KDA-Only`'s published `config.json` carries `use_cache: false` — the trainer's leftover, shipped as-is. That same config has `linear_attn_config.kda_layers` = 1–36 with `full_attn_layers` empty, and **no** `layer_types` field, consistent with them bypassing transformers' hybrid cache machinery entirely — which is what §3 found from the other side.

### Pinning the commit

In `r1_stage1_4node.sbatch`, extend the existing provenance block:

```bash
echo "--- afm_kda (the model code) ---"
git -C $WORK_LUKAS/afm_kda log -1 --oneline
git -C $WORK_LUKAS/afm_kda status --porcelain || echo "(clean)"
```

and optionally, after `srun`, write it beside the weights (`OUTDIR` = whatever `output_dir` is in `r1_stage1_4node.yaml`; the sbatch has no such variable of its own):

```bash
git -C $WORK_LUKAS/afm_kda rev-parse HEAD     > $OUTDIR/afm_kda_commit.txt
git -C $WORK_LUKAS/afm_kda status --porcelain > $OUTDIR/afm_kda_dirty.txt
```

⭐ **The `--porcelain` line is the one that earns its keep.** The commit hash only describes what ran if nothing was edited after the commit. Empty = the hash is trustworthy. Non-empty = it is a lie, and says which files differ.

⚠️ **Which is why `logs/` and `__pycache__/` must be gitignored.** The first commit swept in 55 files of SLURM output; every later test job writes more, so `status --porcelain` would be non-empty on every run and the signal would be worthless.

---

## 7. What remains

### ✅ B7c — one real `generate()` call, PASSED 2026-09-22

Three prompts, left-padded into a batch of width 17 with pads `[11, 9, 0]`, 48 greedy tokens through `model.generate()`.

| row | pads | student output |
|---|---|---|
| "The capital of France is" | **11** | *"…Paris. It is the largest city in France… located in the north-central part of the country, on the Seine River. The city is divided into 20 arrondissements"* |
| "Water boils at a temperature of about" | 9 | *"…100 degrees Celsius. The boiling point of water is the temperature at which the vapor pressure of the liquid equals the atmospheric pressure…"* |
| "In July 1969, three astronauts travelled to the Moon aboard a spacecraft named" | 0 | *"…Apollo 11. The mission was a success…"* |

⭐ **The row carrying the most padding produced the best output.** That is the specific failure this test was written to catch — pad tokens entering the recurrent state would degrade the short, heavily-padded rows while leaving the unpadded one clean — and it is ruled out, not merely absent.

📌 **Do not read the teacher's row 0 as a quality signal.** The teacher looped (*"Paris is the capital of France"* six times) where the student did not. Greedy decoding on a base model, n = 1. It says nothing about either model.

**`config.use_cache` printed `True`**, verified against the loaded model rather than the file on disk.

#### Why it was worth a job of its own

⚠️ **Every test on this track hand-rolled the decode loop.** B6, B6d and B7 all call `m(input_ids=..., past_key_values=pkv)` by hand. lm-eval calls `model.generate()`, which reads `generation_config`, **constructs its own cache**, and manages `cache_position` itself. That code path has never run.

`test_b7c_generate.py`: three prompts of different lengths, left-padded into one batch, 48 greedy tokens, teacher first as control. It also prints `config.use_cache` **as loaded**, verifying the flip against the model rather than against the file on disk.

```bash
sed 's/test_b7_batched/test_b7c_generate/g' $WORK_LUKAS/afm_kda/test_b7_batched.sbatch \
  > $WORK_LUKAS/afm_kda/test_b7c_generate.sbatch
sbatch $WORK_LUKAS/afm_kda/test_b7c_generate.sbatch
```

**Read it for coherence, not for agreement with the teacher.** Greedy decoding is chaotic and B6c settled that the two diverge on near-ties. Pass: fluent, on-topic English on all three rows. Fail: repetition loops, or — the one to watch — the two **short** prompts degenerating while the long one is fine, which would be pad contamination that B7's 32-token comparison was too gentle to catch.

### 🔶 B8 — GSM8K and IFEval: the smoke run, and the control that closed Track B for good

Debug queue, teacher first, `use_cache=True` forced on both.

| | n | GSM8K 5-shot |
|---|---|---|
| **teacher** | 20 | **0.60 ± 0.11** — consistent with Arcee's published 0.52, so the harness is sound |
| **student, cache ON** | 20 | 0.00 strict / 0.05 flexible |
| **student, cache OFF** | **5** | 0.00 / 0.00 — and **byte-identical generated text** on the five it shares |

⚠️ *The cache-OFF control ran with `LIMIT=5`, not 20. It is a control, not a score: what it establishes is that the first five generations are character-for-character identical with and without the cache, which is all it was asked to do. Do not read 0.00/0.00 there as a measurement.*

⭐⭐ **The control is what matters here.** Running the identical prompts through the full-recompute oracle produced the *same characters*, to the truncation point, including the same `#### 150,000` and the same `[invalid]` extractions. Not "agreeing within noise" — identical, despite the cached path switching Triton kernels mid-generation. **Whatever is wrong with the GSM8K number, it is not the cache.** Track B is closed by its own oracle.

**What the samples show.** The student reproduces the few-shot *form* perfectly — `<<16*7=112>>` annotations, the `####` marker, clean termination on one of three — and fails the *arithmetic*: `2 + 0.5 = 2`, `150,000 − 120,000 = 150,000`. Two of three then loop. (The teacher loops too, on B7c's prompts; greedy decoding on a base model does this.)

**Why a model at 79–89 % relative on ARC-C, HellaSwag and WinoGrande reads 4.5 % here.** Held-out CE is 2.4559 against the teacher's 2.2963 — a 17 % perplexity penalty — and a five-step arithmetic chain scored by exact match degrades **multiplicatively** where a single-option choice degrades gracefully.

⚠️⚠️ **That was written as the general account of the benchmark spread, and MMLU falsified it on 2026-09-25.** MMLU is scored by likelihood over four fixed options — the *same* machinery as ARC-C — and it came in at **13.9 % relative** while ARC-C held at 79.1 %. **The line that actually separates the benchmarks is the shot count**: both 5-shot tasks are at the floor, all three 0-shot ones hold. Multiplicative composition survives only as the local reason GSM8K reads 4.5 % rather than 14 %. Full account, and the one job that separates the two surviving mechanisms: `claude/first_distillkit_runs.md` revision 20.

⭐⭐ **And the cause is now measured, not guessed — but it is not what I first wrote here.** `test_r1_ce_math.py` compared the student-vs-teacher CE gap across three corpora:

| corpus | gap | relative to web |
|---|---|---|
| web (FineWeb-Edu — **the training distribution**) | +0.1595 | 1.00× |
| math (MetaMathQA) | +0.3372 | **2.11×** |
| instruction / code (Tulu 3) | +0.3320 | **2.08×** |

**Mathematics is not special. Distance from the training distribution is.** An instruction-and-code corpus, whose first document is Terraform, shows the same doubling. An earlier draft of this section ranked "frozen MLPs" as the best-supported cause; that ranking did not survive the control. **Full account and the arithmetic that rules out Tulu 3's own maths content as an explanation: `claude/first_distillkit_runs.md`, "What stage 1 did NOT recover".**

✅ **The full run landed 2026-09-25 and the caution was vindicated.** Teacher **0.5353 ± 0.0137** strict — within 1.4 points of Arcee's published 52.1, so the harness is sound on the real set too. Student **0.0243 ± 0.0042** strict, **0.0349** flexible.

⭐ **It is not zero.** 0.0243 on 1,319 items is about **32 problems solved**, **5.8 σ above zero**. The smoke run's 0/20 was consistent with a true rate up to ~14 %; the answer sits at the low end of that interval, which is exactly what the caution was for.

⭐ **And the student loses the output FORMAT as well as the arithmetic.** Flexible 0.0349 against strict 0.0243 — roughly fourteen problems where the right number appeared but the `####` marker did not. The teacher is the other way round (strict 0.5353 ≥ flexible 0.5345): it always emits the marker. A model failing to reproduce a format it was shown five times is a **few-shot conditioning** signal, not an arithmetic one.

⚠️ **And do not read the timing as a speed measurement.** The student took ~9 s/item against the teacher's ~2.3 s, but a model that fails to emit the stop string runs to the token cap — the ratio is confounded by output length. B9b is the measurement.

⬜ **IFEval is the one remaining benchmark.** The budget argument all of this feeds is in `claude/first_distillkit_runs.md`, open decision 2.

*Trap: **a modest perplexity penalty is not a modest benchmark penalty.** Errors compose multiplicatively along a generated chain and independently on a multiple-choice question.*

### B9b — decode throughput vs context length ⬜

**B9a measured the memory half of the architectural claim. B9b is the speed half, and it is the one that needed the bridge.**

During decoding an attention layer must attend over **every key it has cached**, so the cost of one more token grows with how much text is already there. A KDA layer multiplies the new token against a **fixed** 20 × 128 × 128 state, so its per-token cost is the same at token 30,000 as at token 30.

```
for T in 512, 1024, 2048, 4096, 8192, 16384, 32768:
    prefill a prompt of T tokens with use_cache=True
    torch.cuda.synchronize(); t0 = time()
    generate 64 tokens one at a time from the cache
    torch.cuda.synchronize()
    report (time() - t0) / 64  ->  seconds per token
```

⚠️ **Discard the first few tokens** — Triton autotuning fires on the first call of each kernel shape and costs minutes. Warm up, then time.

Expected shape: the teacher's curve rises with T; kda_only's is flat; the hybrid's rises at roughly a quarter of the teacher's slope, since it keeps 9 of 36 attention layers. **The crossover will not be where B9a's 670 tokens is** — memory and time have different constants — and finding it by measurement is the point.

Together B9a and B9b are the two halves of the sentence this thesis exists to write: *converting attention to KDA costs this much quality and buys this much memory and this much decode speed, at this context length.*

---

## 8. Scripts

All in `$WORK_LUKAS/afm_kda/` and mirrored in the project folder under `claude/`. Each has an sbatch made by `sed`-renaming a sibling.

| file | what it answers |
|---|---|
| `test_b0_probe.py` | can fla's `Cache` replace transformers' wholesale? (no) |
| `test_b9_cache_memory.py` | inference memory across seven context lengths, measured and cross-checked |
| `test_b6_cache_equiv.py` | cached vs uncached vs `generate()`, token for token; measures the fla state |
| `test_b6d_precision.py` | teacher-forced, in bf16 and fp32, with the teacher as control |
| `test_b7_batched.py` | batched, left-padded, cached, against per-row unpadded references |
| `test_b7b_mask_probe.py` | prints per-layer cache answers and the mask's shape/stride/contiguity |
| `test_b7c_generate.py` | one real `generate()` call |
| `finalize_checkpoint.py` | verify markers + `auto_map`, flip `use_cache`, report the git stamp |

⚠️ **`from test_b9_cache_memory import cache_bytes` runs the whole benchmark** — no `__main__` guard — which is why a B9 table once appeared inside B6's output. B6 carries its own copy of the function. *An import is a program you are running, not a symbol you are borrowing.*

---

## 9. Traps this track produced

**On diagnosis**

- **⭐ Read the library before patching around it.** transformers 5.8.1 already implements the hybrid `get_mask_sizes` redirect; it just could not see our layers. Four greps on a login node, during a scheduler outage, replaced a week of guessing and produced a four-line fix.
- **⭐ A control beats a tolerance.** Running the unmodified teacher through the identical protocol answered in one line what three invented thresholds could not: the student's 1.875e-01 *is* the teacher's 1.875e-01.
- **⚠️ An absolute floating-point tolerance is meaningless without a magnitude.** One bf16 ulp at logit scale 34.5 is **0.25**; a "1e-2 tolerance" was below the representable resolution and made a clean pass look like a 15× failure.
- **⚠️ Read the kernel before trusting a tolerance.** fla runs `chunk_kda` above 64 tokens and `fused_recurrent_kda` at or below, so "the same computation" through a cache is literally a different program.
- **⚠️ Change the precision to separate rounding from a bug.** Rounding shrinks by ~2¹⁶ from bf16 to fp32; a logic error does not shrink at all. No threshold required.
- **⚠️ "Does the error grow?" beats "is the error small?"** Flat per-position error is rounding; accumulating error is broken state. The first question needs no constant.
- **⚠️ "fp32" on an Ampere GPU is often TF32.** 10-bit mantissa, ε ≈ 4.9e-4.
- **⚠️ Separate what you observed from what you inferred.** B0's failure is observed; its cause is not, and acting on the guess would have discarded the simpler design for nothing.

**On testing**

- **⚠️ A pass under a configuration that bypasses the patched code is not evidence about the patch.** `flash_attention_2` short-circuits `create_causal_mask`, so B7's clean fa2 rows said nothing about the padding fix. Ask which run *could* have failed.
- **⭐ The slow oracle keeps earning its keep.** `use_cache=False` was B1's ground truth, and months later it was the one experiment that could exonerate the cache against a bad benchmark number. Keep a correct-but-slow path available forever.
- **⚠️ `--limit N` is a pipeline test, not a measurement.** 0/20 is consistent with a true rate up to ~14 %, and a generation-time ratio from it is confounded by output length.
- **⚠️ A modest perplexity penalty is not a modest benchmark penalty.** +0.16 nats coexists with 74–89 % relative retention on discriminative tasks and ~0 on a multi-step generative one, because errors compose multiplicatively along a chain.
- **⚠️ A fallback that nothing consumes cannot be tested by running it.** In an all-linear model `get_seq_length` feeds only RoPE and `get_mask_sizes` feeds only a causal mask, and no linear layer reads either — so a permanently-zero counter passes a green run. **Ask what would *observe* a wrong value before calling a pass coverage.**
- **⚠️ A threshold printed by your own script ages.** `test_b6d_precision.py`'s verdict text still asked for 1e-5 long after TF32 had moved the real floor to ~1e-2 — it would have failed the validated hybrid. Read a decision rule against the reference *measurement*, not against the number you wrote before you had one.
- **⚠️ A hand-rolled loop is not the code path the harness uses.** Five tests validated a decode loop we wrote; none called `generate()`. B7c closes that.
- **⚠️ Greedy decoding is a chaotic map, so it is a bad equivalence test.** Two implementations agreeing to within a bf16 ulp still produce different text at the first near-tie. **Teacher-force the comparison** and the expected answer becomes exact.
- **⚠️ A "logit difference" measured after a divergence is meaningless.** 2.178e+01 was two different contexts being compared, not an error magnitude.
- **⚠️ A test that crashes has not failed — it has not run.** Order the subjects so the control is reached first, and catch per-subject.
- **⚠️ Two paths can agree because both are right, or because the test cannot tell them apart.** `assert fla_mb > 0` is what makes B6's agreement mean something.
- **⚠️ A discriminating test needs an input the two branches can disagree on.** Sixteen arbitrary token ids hid a real bug behind a 4 % difference. Use real text, a few hundred tokens.
- **⚠️ Batch + padding + cache is a combination five earlier tests never exercised together.** Each was individually covered; the interaction was not.
- **⚠️ Check the arithmetic of your own probe.** B7b printed `600` everywhere because it passed `cache_position` where `query_length` belonged. The comparison survived; the number did not.
- **⚠️ Name a check once.** B6 / B6b / B6c drifted between labels across three messages and cost an hour of not knowing what had passed.

**On the code and the machinery**

- **⚠️ When a class answers one question per layer, check whether it answers others too.** Fixing `get_seq_length` and stopping there left an identical bug one method along in `get_mask_sizes`.
- **⚠️ A cache can lie about its length.** `get_seq_length()` reads layer 0, and in a hybrid layer 0 stores nothing.
- **⚠️ A broadcastable wrong shape does not raise.** A mask of width 1 against 301 keys broadcasts silently; the error surfaces three layers down as a contiguity complaint about a stride the kernel created itself. **Where an error is raised is not where it was caused.**
- **⚠️ Preventing a crash is not fixing a bug.** Dropping the 4-D mask kept fla's assertion quiet and let pad tokens into the recurrent state instead. A holding move has to be logged as one.
- **⚠️ A simpler implementation may be simpler because it supports less.** The reference generates easily because it never builds a causal mask — exactly the defect that makes it wrong under sdpa or eager.
- **⚠️ A fixed-size state is not a free state.** 1.31 MB per KDA layer regardless of context, so a converted model is *heavier* than its teacher below T ≈ 670. Quote the crossover, not just the asymptotic ratio.
- **⚠️ Infer a dtype from byte counts only as a hypothesis, then look.** The recurrent state is fp32 and the conv state is `conv_size` wide; both assumptions were wrong and the error was 2×.
- **⚠️ When a closed-form check starts failing, ask what the measurement now includes.** B9a's MISMATCH was the check noticing the cache had begun carrying KDA state as well as KV.
- **⚠️ Count the layers before quoting them.** "31 layers, 22 KDA" was `hybrid31` misread as a count. It is 36 layers, 27 KDA, 9 attention, 3:1.

**On the workflow**

- **⚠️ Fix the chain, not the artefacts.** Copying `modeling_afm_kda.py` into every checkpoint is correctness by remembering; committing `afm_kda` and rebuilding the init students is correctness by construction, and `save_pretrained` does the rest.
- **⚠️ A checkpoint's `.py` files are frozen at the moment the run started.**
- **⚠️ A signal that always fires is no signal.** An un-ignored `logs/` directory makes the dirty-tree check permanently red.
- **⚠️ The module cache lives under `$HF_HOME`.** Clearing `~/.cache/huggingface` is silent and does nothing.
- **⚠️ An import is a program you are running, not a symbol you are borrowing.** Guard scripts with `if __name__ == "__main__":`, or copy the function.
- **⚠️ `traceback.print_exc(limit=N)` truncates from the OUTSIDE.** It hid the raising frame twice on this track. Use no limit.
- **⚠️ stdout and stderr are two different files under SLURM.** `traceback.print_exc(file=sys.stdout)`.
- **⚠️ A variable assigned inside an sbatch script beats `--export`.** Use `${VAR:-default}`.
- **⚠️ Check what a guard sees on every path that reaches it, including evaluation.** A `use_cache` check written for training would have fired on R1's first eval and killed a four-hour run.
- **⚠️ A ten-minute probe can retire a day of work.** B0 cost one debug job and killed an approach that would have taken three steps to discover was impossible.
- **⚠️ `X.__dict__` answers "defined here", not "defined by this library".**

---

## Sources

- `kda_installed_0.5.2.py` — the authoritative fla source (line 211: the kernel switch; the unpad/repad block)
- `modeling_arcee_kda.py` — the reference; ⭐ read for what it *avoids*, not only what it does
- transformers 5.8.1, read directly: `cache_utils.py` (`DynamicCache.__init__`, `get_mask_sizes`, `LAYER_TYPE_CACHE_MAPPING`), `masking_utils.py`, `integrations/sdpa_attention.py`, `models/qwen3_next/` as a working hybrid in the same version
- [arcee-ai/AFM-4.5B-Base-KDA-Only](https://huggingface.co/arcee-ai/AFM-4.5B-Base-KDA-Only) — `use_cache: false` in the published config
- [Distilling Kimi Delta Attention into AFM-4.5B](https://www.arcee.ai/blog/distilling-kimi-delta-attention-into-afm-4-5b-and-the-tool-we-used-to-do-it), Charles Goddard