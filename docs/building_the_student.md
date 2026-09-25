# Track A — Building the KDA Student ✅ **COMPLETE**

*Companion to `claude/startup_plan_and_pitfalls.md` (step 7 there points here) and `claude/first_distillkit_runs.md` (**the live document**). This one covers the model itself.*

*Revision 32 (2026-09-25) — ⚠️⚠️ **"What stage 1 does and does not recover" was wrong and is rewritten.** The full MMLU run — teacher **0.6500**, student **0.3055**, **13.9 % relative** — falsifies the likelihood-against-generation split this section was built on: MMLU is scored by likelihood over four fixed options, exactly like ARC-C, and it collapsed while ARC-C held at 79.1 %. **The dividing line is the shot count**: both 5-shot benchmarks are at the floor, all three 0-shot ones hold. Numbers, controls and the one job that separates the two surviving mechanisms: `claude/first_distillkit_runs.md`, revision 20.*

*Revision 31 (2026-09-22) — ⭐ **Track B is CLOSED.** The student generates: batched, left-padded, cached, through `generate()`. Two defects were found and fixed (mask width from a KDA layer; the 2-D padding mask never reaching KDA) — full account in `claude/track_b.md`. The Track B section below shrinks to its result. ⚠️ Also **deletes the superseded B9a table**: revision 27's 594 MB / 24.1 MB / 95.7× figures were computed under a bf16 assumption and were corrected by measurement in revision 28 to **611 MB / 47.1 MB / 48.9×**. Both tables were present and the wrong one was quotable. Track C's "do not interleave with Track B" constraint is lifted.*

*Revision 30 (2026-09-20) — ⚠️ **the memory numbers are corrected by measurement.** Once B4' made the KDA layers actually store state, B6b measured **35.33 MB** for 27 layers where I had computed 18.06: the recurrent state is kept in **float32**, and the conv state is `conv_size` wide. The per-layer constant is **1.3086 MB**, so at 32k the hybrid is **3.77x** and kda_only **48.9x**. ⭐ And the fixed state is not free: below a crossover at **T = 670 tokens** a converted model carries MORE state than the teacher.*

*Revision 29 (2026-09-20) — **Track B moves out of this document** into `claude/track_b.md`, which holds the whole plan, the patch and the scripts in one place; what remains here is a one-screen summary.*

*Revisions 16–28 (2026-09-18 → 09-20) — B9a measured and corrected; B0/B0b closed and the fla-Cache-everywhere design abandoned; the "100× smaller" headline corrected to a per-layer number; `build_info.json` verified; A7c and A9 closed; four students built from `3aafffc`.*

---

## 🗺 The three tracks

| Track | What it is | State |
|---|---|---|
| **A** | Build the KDA student and prove it is the model we think it is | ✅ **complete** — `3aafffc`, four students, R1 trained |
| **B** | Give it a working recurrent cache so it can **generate** | ✅ **complete** — `d7cd0a0`; full account in `claude/track_b.md` |
| **C** | Build the **GDN** sibling student for the mechanism comparison | ⬜ **patch written**, full plan in `claude/building_GDN_student_track_c.md` |

Track B gated GSM8K, IFEval, and every inference-side measurement; those are now ordinary evaluation jobs. Track C gates the comparison that makes the thesis a study rather than a replication.

---

## ⭐ How the student was built — the procedure

**The one-sentence version:** Arcee published a converted model (`modeling_arcee_kda.py` + `configuration_arcee_kda.py`, written against transformers 4.57) but **not** the script that converted it; we read their two files to learn the *architecture*, then wrote **our own** `modeling_afm_kda.py` + `configuration_afm_kda.py` by **subclassing the current library's Arcee implementation** instead of copying theirs — and verified every step against the teacher.

| Step | What we set out to do | What we actually did | Artifact produced | How we knew it worked |
|---|---|---|---|---|
| **A0** | Find out what the target architecture *is* | Read Arcee's two files and the blog post; read `fla/layers/kda.py` | layer pattern, NoPE, the undocumented attention gate, the 5× GQA repeat, **six defects** | blog post and code agree; each defect re-tested later |
| **A1** | Get a *baseline* we control | Copied the **library's** Arcee files (transformers 5.x) into `$WORK_LUKAS/afm_kda`, absolutised their imports, committed them **unmodified** | `afm_kda/` git repo + `TRANSFORMERS_VERSION` | imports resolve; geometry matches `config.json`; predicted parameter budget |
| **A2** | Write the student **without** copying Arcee's file | `configuration_afm_kda.py` (`@strict` dataclass subclass, three flags) and `modeling_afm_kda.py` (subclasses + a thin **adapter** wrapping `KimiDeltaAttention`) | our two `.py` files | — verified by A3 onward |
| **A3** | Prove we broke nothing | Ran the student with **all three flags at AFM's defaults** against the teacher | `test_a3.py` | **logit diff 0.000e+00**, twice. Caught two initialization bugs. |
| **A3b** | Price the two architecture changes separately | Flipped `use_rope` off, then `use_attn_gate` on, one at a time | `test_a4.py` | NoPE **+3.68 CE**, gate **+0.58 CE** — thesis numbers |
| **A4 / A5** | Get KDA layers to actually run | 1 KDA layer, then all 27; forward **and** backward | `test_a4.py` | grads 624/624, bf16, **4.9990 B** = prediction |
| **A5b** | Transfer the teacher's q/k/v/o into KDA | Reimplemented the teacher's own `repeat_kv` (group-major) | `transfer_attention_into_kda()`, `test_a5b.py` | 20/20 head mappings; **−0.21 CE**; **CE 7.9412 — later confirmed by R0** |
| **A6** | Find out what KDA actually buys | One KDA layer vs one attention layer, T = 2k … 32k | `test_a6_curve.py` | crossover ~16k; **0.31× at 2k** — corrected twice |
| **A7** | Prove the KDA layer honours `cu_seqlens` | Packed two documents, passed `cu_seqlens` **by hand** | `test_a7.py` | **0.000e+00** — perfect reset; checkpointing irrelevant |
| **A7b** ⭐ | Does a `cu_seqlens` **arrive** at runtime? | Same comparison through the **packing mask** | `test_a7b.py`, `build_cu_seqlens()` | **FAILED at 3.125e+00**, then **0.000e+00** after the fix |
| **A7c** ⭐ | And at the **9 attention layers**? | Same on `hybrid31` under `flash_attention_2`; then **counted** the monkey patch's calls | `test_a7c.py`, `test_a7c_probe.py` | **FAILED at 1.385e+00**; patch fires **zero times**; **0.000e+00** after the varlen fix |
| **A8** | Make it loadable the way DistillKit loads it | Saved, then reloaded **in a fresh process** via `AutoModelForCausalLM` + `trust_remote_code` | `build_student.py` → four students | `AFMKDAForCausalLM`, 4.999026076 B |
| **A9** 📌 | Could the **teacher** ever honour a segment mask? | A7c's test with `arcee-ai/AFM-4.5B-Base` in place of the student | `test_a9.py` | **No — 1.302e+01 for both the segment mask and the control.** Closes the question |

### Why not just copy Arcee's two files?

| | copy theirs | subclass the library (**ours**) |
|---|---|---|
| transformers version | written for **4.57**; we run **5.x** | matches what we run |
| causal mask | never built (their defect) | inherited `create_causal_mask` — ✅ A3 |
| gradient checkpointing | hand-rolled loop that drops `**kwargs` | inherited `GradientCheckpointingLayer` — ✅ A7 |
| packed boundaries, **KDA** | raw mask reaches KDA — but fla **can't split on it** | `cu_seqlens` built explicitly — ✅ A7b |
| packed boundaries, **attention** | flash's unpad path is never reached, so the patch is inert | varlen kwargs injected explicitly — ✅ A7c |
| **generation / KV cache** | ✅ works — they unified on fla's `Cache` | ✅ **built — Track B, `d7cd0a0`** |
| lines we own | ~500 | ~200 + Track B's ~40 |
| ablations | edit the file | one config flag |

⚠️ **Revision 16 got the KDA row backwards** and was corrected in rev 17. Neither implementation got document isolation for free.

**The honest scorecard, now that Track B has landed:** our choice bought training correctness and a fifth of the code; theirs bought a complete model including generation. Track B was the bill for that ordering, it came to about forty lines, and **what we ended up with is strictly more capable than the reference** — correct under sdpa, eager *and* flash-attention, where theirs is correct only under flash-attention because it never builds a causal mask. **Neither ordering was wrong; ours optimised for the right thing first and the debt was payable.**

---

## Where the student lives

| Thing | What it is | Where |
|---|---|---|
| **DistillKit** (your fork) | the *trainer* | `$WORK_LUKAS/distillkit` |
| **AFM-4.5B-Base** | the *teacher*, unmodified | `$HF_HOME/hub/models--arcee-ai--AFM-4.5B-Base` |
| **Student source** | the model *definition* + tests + docs | `$WORK_LUKAS/afm_kda` (git, **`d7cd0a0`**) |
| **Student checkpoint** ⬅ what the YAML points at | built, saved, reload-verified | **`$WORK_LUKAS/students/afm_kda_hybrid31_init`** |
| **Trained checkpoint** | R1, stage 1, 300 M tokens | `$WORK_LUKAS/output/r1_stage1_hybrid31` |
| *(reference)* | Arcee's converted student — read, don't copy | `$WORK_LUKAS/reference_kda_nope` |

```
afm_kda/                       (git repo -- code only)
  __init__.py                  (empty -- makes the relative imports resolve)
  configuration_arcee.py       library copy, imports absolutised   } baseline
  modeling_arcee.py            library copy, imports absolutised   }  commit
  TRANSFORMERS_VERSION
  configuration_afm_kda.py     yours
  modeling_afm_kda.py          yours   (+ Track B's cache bridge)
  build_student.py             builds + saves a student  (--variant)
  finalize_checkpoint.py       verify markers, flip use_cache       <- Track B
  test_a*.py  / .sbatch        the Track A ladder
  test_b*.py  / .sbatch        the Track B ladder
  eval_generative_afm.sbatch   GSM8K + IFEval
  docs/                        the four project documents, versioned
  .gitignore                   __pycache__, *.safetensors, logs/
```

⚠️ **Keep the checkpoints OUT of the git repo.** ~10 GB of safetensors inside `afm_kda/` would make `git add -A` unusable — hence the separate `students/` directory.

⚠️ **`logs/` must be gitignored too.** Without it, `git status --porcelain` is non-empty after every test job, which makes the provenance dirty-check permanently red and therefore useless.

⚠️ **Editing `afm_kda/` does not change a checkpoint.** Each `students/*_init/` directory holds its own copies of the five `.py` files, and `from_pretrained` imports *those*.

⭐ **The provenance model, settled 2026-09-22.** Rather than copying fixes into checkpoints after the fact — correctness by remembering — the chain is:

```
afm_kda/ under git  ->  students/*_init built from it  ->  save_pretrained carries
the .py files into output/<run>  ->  the commit hash identifies the code
```

`finalize_checkpoint.py` therefore **verifies** rather than copies: it asserts a list of markers is present in the checkpoint's own `modeling_afm_kda.py`, checks `auto_map`, flips `use_cache`, and reports the git stamp. `--sync` (copy) is opt-in and exists only to retrofit a pre-fix checkpoint; **needing it on a fresh run is itself a finding.**

### The four students ✅ *(all built from `3aafffc`, 2026-09-18; rebuild from `d7cd0a0` pending)*

| Variant | Directory | What defines it |
|---|---|---|
| **hybrid 3:1 — the main run** | `students/afm_kda_hybrid31_init` | 27 KDA layers, `use_rope: false`, `use_attn_gate: true` |
| KDA-only | `students/afm_kda_only_init` | all 36 layers KDA — **the clean test bed for A7b**, and the most valuable second run |
| FLP | `students/afm_kda_flp_init` | full attention at 0,1,2,3 + 5 more — ⚠️ **layout not confirmed against theirs** |
| de-novo ablation | `students/afm_kda_hybrid31_denovo_init` | same as hybrid31, `--no-transfer` (RADLADS's ablation) |
| *(Track C, planned)* | `students/afm_gdn_hybrid31_init` | Gated DeltaNet — see Track C below |

Convention: `afm_<mechanism>_<layout>_<stage>` — `init` before training, the run name after.

⚠️ **The directory name is only a label.** What defines the model is `config.json`.

⚠️ **Before a `--force` rebuild, hash the weights first.** If the build is not deterministic, overwriting `afm_kda_hybrid31_init` destroys R1's exact starting point:

```bash
cd $WORK_LUKAS/students
sha256sum afm_kda_hybrid31_init/*.safetensors > ~/init_before.txt
# --- the real builder, with --force ---
sha256sum -c ~/init_before.txt
```

`OK` means the init is reproducible from the commit. A mismatch is not a problem — it means the init is seeded per-run and the thesis says so — but you want to know which.

### ✅ `build_info.json` — verified 2026-09-18

```json
{
  "variant": "hybrid31",
  "teacher_parameters": 4619184640,
  "parameters": 4999026076,
  "config": { "kda_layers": [0,1,2,4,...,34], "full_attn_layers": [3,7,...,35],
              "num_kda_layers": 27, "use_rope": false, "use_attn_gate": true },
  "attention_transferred": false,
  "seed": 0,
  "loading_info": { "missing_keys": 441, "unexpected_keys": 108 },
  "afm_kda_git": { "commit": "3aafffcaa3b7...", "short": "3aafffca",
                   "branch": "master", "dirty": true },
  "versions": { "python": "3.11.7", "torch": "2.12.0+cu126",
                "transformers": "5.8.1",
                "flash-linear-attention": "0.5.2", "fla-core": "0.5.2" },
  "slurm_job_id": "58107463", "host": "lrdn0068.leonardo.local",
  "code_files": ["__init__.py", "configuration_afm_kda.py", "modeling_afm_kda.py",
                 "configuration_arcee.py", "modeling_arcee.py"]
}
```

*(This example is the de-novo build, hence `attention_transferred: false`.)*

⭐ **`loading_info` is an independent check on the entire initialization inventory, and it lands exactly.**

- **unexpected = 108 = 27 × 4.** The teacher's `q/k/v/o_proj` in each KDA layer have no name to land on — ours are `self_attn.attn.*`, one level deeper. `transfer_attention_into_kda()` copies them in by hand (A5b).
- **missing = 441 = 27 × 16 + 9.** Sixteen new tensors per KDA layer, plus the nine attention `gate_proj`s.

Both are exact, and together with `parameters == 4,999,026,076` they pin the build three independent ways.

⚠️ **These four builds record `"dirty": true`.** The five `code_files` copied into each checkpoint are the committed ones, so the models are honest, but a dirty flag means the hash alone does not reconstruct the tree. **Commit before every build and run** — and the rebuild from `d7cd0a0` should record `dirty: false`.

### ⭐ Iterating on model code without rebuilding

```bash
cd $WORK_LUKAS
for V in hybrid31 hybrid31_denovo only flp; do
  D=students/afm_kda_${V}_init
  cp afm_kda/{__init__.py,configuration_afm_kda.py,modeling_afm_kda.py,configuration_arcee.py,modeling_arcee.py} $D/
  rm -rf $HF_HOME/modules/transformers_modules/afm_kda_${V}_init
done
```

Seconds instead of thirty minutes, because a forward-logic fix leaves the weights bit-identical (`torch.manual_seed(0)`, same init path). **Fast loop for debugging; a full `--force` rebuild afterwards** so `build_info.json` records the hash of the code actually in each checkpoint.

⚠️ **Do not run this while a job is queued or running** unless you intend that job to use the new code.

⚠️ **The re-sync must cover the TRAINED checkpoints too**, not only `students/` — `output/r1_stage1_hybrid31` and `output/r1_stage1_hybrid31/checkpoint-4580` carry their own copies, and a test pointed at either would otherwise measure the pre-fix code. This bit us once during Track B.

### ⚠️ `$HF_HOME/modules/transformers_modules/` is not the student

It is a **code cache**: transformers copies the `.py` files out of a remote-code checkpoint into there, keyed by content hash. A few hundred KB, no weights. **Always safe to delete** — and you *must* delete it after changing any `.py` or renaming a checkpoint.

---

## ⭐ The complete initialization picture

### The skeleton — identical for both layer types

```
AFMKDAForCausalLM
├── model.embed_tokens             [128004 × 2560]   ← teacher
├── model.layers[0 … 35]
│   ├── input_layernorm            [2560]            ← teacher
│   ├── post_attention_layernorm   [2560]            ← teacher
│   ├── mlp.up_proj                [18432 × 2560]    ← teacher
│   ├── mlp.down_proj              [2560 × 18432]    ← teacher
│   └── self_attn                  ← THE ONLY PART THAT DIFFERS
├── model.norm                     [2560]            ← teacher
└── lm_head                        [128004 × 2560]   ← teacher
```

Everything outside `self_attn` is the teacher's, byte for byte. RADLADS's point made concrete: **factual knowledge lives in the MLPs and embeddings; you are only replacing the routing machinery.**

⚠️ **And stage 1 freezes the MLPs**, which is the single most likely explanation for the one capability that did not survive — see "What stage 1 does and does not recover" below.

### The 9 attention layers — 0-based indices 3, 7, 11, 15, 19, 23, 27, 31, 35

```
self_attn  (AFMKDAAttention)
├── q_proj      [2560 × 2560]   ← teacher, direct
├── k_proj      [ 512 × 2560]   ← teacher, direct   (narrow: GQA, 4 KV heads)
├── v_proj      [ 512 × 2560]   ← teacher, direct
├── o_proj      [2560 × 2560]   ← teacher, direct
└── gate_proj   [2560 × 2560]   ★ NEW, random       (only if use_attn_gate)
```

📌 **These nine layers are where Track B did most of its work.** They are the only ones that touch `past_key_values` through transformers' API — and both of Track B's defects lived in how that API answers questions about them.

### The 27 KDA layers — everywhere else

```
self_attn  (AFMKDALinearAttention — our adapter)
└── attn  (KimiDeltaAttention — fla's layer)
    ├── q_proj        [2560 × 2560]   ← teacher, renamed          (A5b)
    ├── k_proj        [2560 × 2560]   ← teacher, 5× head repeat   (A5b)
    ├── v_proj        [2560 × 2560]   ← teacher, 5× head repeat   (A5b)
    ├── o_proj        [2560 × 2560]   ← teacher, renamed          (A5b)
    ├── q_conv1d      [2560 × 1 × 4]  ★ NEW   short convolution
    ├── k_conv1d      [2560 × 1 × 4]  ★ NEW
    ├── v_conv1d      [2560 × 1 × 4]  ★ NEW
    ├── f_proj.0      [ 128 × 2560]   ★ NEW   FORGET gate, per-CHANNEL via a bottleneck
    ├── f_proj.1      [2560 ×  128]   ★ NEW
    ├── g_proj.0      [ 128 × 2560]   ★ NEW   OUTPUT gate, also a bottleneck
    ├── g_proj.1      [2560 ×  128]   ★ NEW   (+ bias [2560] -- a SIXTEENTH tensor)
    ├── b_proj        [  20 × 2560]   ★ NEW   per-head beta
    ├── A_log         [  20]          ★ NEW   decay rate      (special init)
    ├── dt_bias       [2560]          ★ NEW   time-step bias  (special init)
    └── o_norm        [ 128]          ★ NEW   gated RMSNorm, init to ONES
```

**Sixteen tensors per KDA layer**, which is what `missing_keys: 441 = 27 × 16 + 9` confirms.

📌 **Note the two bottlenecks.** Both gates squeeze through 128 dimensions: `f_proj` is 2560→128→2560 and so is `g_proj`. That is a deliberate KDA design choice and it is where most of the parameter difference against Gated DeltaNet comes from — see Track C.

✅ **Verified against the installed fla 0.5.2.** Shapes, names, counts and the initialization scheme are version-independent; only the `cu_seqlens` plumbing differs.

### Why the adapter exists

`ArceeDecoderLayer` does `hidden_states, _ = self.self_attn(...)`. Four things are incompatible in that one call:

| | `ArceeAttention` (what the caller expects) | `KimiDeltaAttention` (what fla gives) |
|---|---|---|
| return values | 2 | 3 |
| `position_embeddings` | required argument | doesn't accept it |
| `attention_mask` | 4-D causal mask from `create_causal_mask` | asserts 2-D |
| `past_key_values` | transformers' `DynamicCache` | **fla's own `Cache`** |

Changing the *caller* means rewriting the decoder layer and the model — the reference's ~500 lines. Changing the *callee* is 40 lines. We did the second, which is why `ArceeModel.forward` and `ArceeDecoderLayer.forward` are inherited untouched.

⚠️ **Row three is how A7b's bug happened, and it came back as Track B's second defect** — the 4-D mask being dropped meant pad tokens entered the recurrent state. ⚠️ **Row four was the whole of Track B**, solved by `AFMKDACache(DynamicCache)` carrying an fla cache as an attribute rather than by replacing transformers' cache.

### The proportions

| | parameters | share |
|---|---|---|
| From the teacher | ~4.90 B | **98%** |
| New, random | ~97 M | **2%** |

**Only 2% of the student is new** — but that 2% is the entire mechanism by which tokens exchange information, which is why the model scores CE 7.94 instead of 2.30 at initialization.

### How the ★ parameters are randomized

- **Ordinary — `normal(0, 0.02)`**: the three short convolutions, `f_proj.0/1`, `g_proj.0/1`, `b_proj`. **Biases zero; `o_norm.weight` ones.**
- **`A_log` — `log(uniform(1, 16))`.** The forget gate is `g = −exp(A_log) · softplus(f_proj(x) + dt_bias)`, so `exp(A_log)` spreads over [1, 16] across the 20 heads — **a deliberate spread of memory timescales.**
- **`dt_bias` — inverse-softplus of a log-uniform draw** in [0.001, 0.1]. Standard Mamba/SSM initialization.

**Both special cases match Arcee's `_init_weights` exactly.** Only the *delivery* differs: they install it as a hook, we apply it to `missing_keys` after loading. `build_info.json` records `seed: 0`.

### The order of operations

```
1. AFMKDAConfig.from_pretrained(TEACHER, kda_layers=..., use_rope=..., use_attn_gate=...)
2. AFMKDAForCausalLM.from_pretrained(TEACHER, config=cfg, output_loading_info=True)
     → copies every parameter whose NAME matches; reports 441 missing / 108 unexpected
3. init_new_parameters(model, missing_keys)      → fills the ★ parameters, nothing else
4. transfer_attention_into_kda(model, teacher)   → KDA q/k/v/o, 5× repeat on k and v
5. save_pretrained + copy the .py files          → the untrained student
```

---

# 🔶 Track B — Generation ✅ **CLOSED 2026-09-22**

*Full account, mechanism, tests and traps: **`claude/track_b.md`**. This is the one-screen summary; do not keep two copies of the steps.*

**The problem.** The student could score text but not write it, and the failure was silent. Our adapter hardcoded `past_key_values=None, use_cache=False`, so the twenty-seven KDA layers kept no state; continuing from a cache would have had the attention layers right and the KDA layers behaving as if the text had just started — fluent, wrong, no error.

**The design.** B0 killed the reference's approach in one debug job: handing fla's `Cache` to the inherited `ArceeModel.forward` raises inside transformers' own wrapper on 5.x. So transformers only ever sees a `DynamicCache` subclass — **`AFMKDACache`** — and the recurrent state rides along as an attribute.

**The two defects, both confirmed by reading the library and then measured.**

| | what was wrong | why it was invisible for so long |
|---|---|---|
| **1. mask width** | `get_mask_sizes` answered from layer 0, a KDA layer that stores nothing, so the causal mask was built **one column wide**. Broadcast to the real key length it acquires **stride 0**, which SDPA rejects | only reachable when a *padding* mask forces a 4-D mask to be built — every earlier test passed no mask at all |
| **2. padding mask** | `create_causal_mask` turns the 2-D padding mask 4-D; fla asserts 2-D; our adapter dropped it — so **pad tokens entered the recurrent state** | the unpadded row always looked fine |

⭐ **transformers 5.8.1/5.9.0 already implements the fix generically** — `LAYER_TYPE_CACHE_MAPPING` marks linear layers as non-`CacheLayerMixin` and `get_mask_sizes` skips to the first attention layer. It could not fire for us because our KDA layers are registered as *nothing*, so lazy replication fills those indices with empty `DynamicLayer`s that look like ordinary attention layers.

**Where it landed.**

| | |
|---|---|
| B5 | ✅ A3 still `0.000e+00` — the training path is untouched |
| B6 / B6e | ✅ cached == uncached, token for token; **the teacher disagrees with itself by exactly the same 1.875e-01**, so the residual is this model's bf16 floor, not our bug |
| B7 / B7b | ✅ batched, left-padded, cached — student 1.1–2.5 ulp against teacher 1.2–1.8, and the **most-padded row is the cleanest** |
| B7c | ✅ one real `generate()` call — fluent, factually correct prose on all three rows |
| B9a | ✅ memory measured: at 32k teacher 2,304 MB, hybrid **611 MB (3.77×)**, kda_only **47.1 MB (48.9×)** |
| B8 | 🔶 GSM8K + IFEval — smoke run done, full runs pending |
| B9b | ⬜ decode throughput vs context length — the headline speed number |

⚠️ **`use_cache` is `false` in every checkpoint the Trainer writes** (gradient checkpointing forces it, `save_pretrained` serialises it) — **and Arcee ship theirs that way too**. Pass `use_cache=True` explicitly in `model_args` for *both* teacher and student; a config flipped on one side only is a fabricated speed comparison.

---

## ⭐ What stage 1 does and does not recover — the GSM8K finding

*Detail and the budget argument: `claude/first_distillkit_runs.md`. Recorded here because it is a statement about the architecture, not about the run.*

After 300 M tokens of stage 1 with MLPs frozen, held-out CE is **2.4559** against the teacher's **2.2963** — a **17.3 % perplexity penalty**, and 97.2 % of the initialization gap closed. That modest language-modelling penalty shows up very differently depending on the task:

⚠️⚠️ **An earlier revision of this section split the benchmarks by scoring mode — likelihood against free generation — and called the asymmetry "mechanical". The full MMLU run falsified it on 2026-09-25**, and the corrected table is below. MMLU is scored by likelihood over four fixed options, exactly like ARC-C, and it collapsed.

| benchmark | **shots** | how it is scored | R1 student, relative |
|---|---|---|---|
| HellaSwag | **0** | likelihood over fixed options | **89.2 %** |
| WinoGrande | **0** | likelihood over fixed options | **79.6 %** |
| ARC-Challenge | **0** | likelihood over fixed options | **79.1 %** |
| **MMLU** | **5** | likelihood over fixed options | **13.9 %** |
| **GSM8K** | **5** | free generation, exact match | **4.5 %** |

⭐⭐ **The line that separates them is the shot count, not the scoring mode.** Both collapsed benchmarks are the only 5-shot ones; all three that held are 0-shot. A 5-shot prompt asks the model to hold five worked exemplars in context and condition on them — **retrieval over distance**, which is what a fixed-size recurrent state is expected to do worst, and what the `MIN_DOC_TOK=600` control already pointed at.

⚠️ **Two mechanisms remain confounded**: few-shot in-context learning itself, and the *distance of a templated 5-shot prompt from raw web text*. Every 5-shot task in the protocol is templated and every 0-shot task is natural prose, so five numbers cannot separate them. **ARC-C at 5-shot does, in one job.**

📌 **Multiplicative composition survives, but only as a local explanation for GSM8K** — why it reads 4.5 % rather than 14 %. The generated samples still show the student reproducing the few-shot **form** (`<<16*7=112>>`, the `####` marker) and failing the **arithmetic** (`2 + 0.5 = 2`). It is no longer the general account.

**It is not a generation bug.** The identical prompts run with `use_cache=False` — the full-recompute oracle — produce **byte-identical output**. Track B is exonerated by its own control.

**The plausible causes, in order of evidential support:**

1. **Budget and frozen MLPs.** Ours is 23 % of Arcee's tokens and omits the stage in which their MLPs unfreeze. RADLADS states outright that freezing MLPs and embeddings hurts. Arithmetic is the capability most associated with MLP circuits, and it is the one that vanished.
2. **Compounding.** See above — this is why the number is ~0 rather than ~15.
3. **The corpus.** FineWeb-Edu is raw web text; Arcee's mix is curated instruction data, which plausibly contains chain-of-thought arithmetic. This is the *same* single difference that explains our KL ≈ 0.35 against their ≈ 0.2 — every other hyperparameter is identical.

⚠️ **GSM8K is the most fragile task even in the reference's own numbers.** Their hybrid retains 70.1 % relative on GSM8K against 93.0 % on HellaSwag; their kda_only drops to 51.4 %. Conversion hurts math most, with four times the budget.

⚠️ **And the consequence for the thesis is about discrimination, not absolute score.** A benchmark that reads ~0 for *every* student cannot separate KDA from GDN — which loses the axis where linear-attention conversions are most often reported to suffer. That is the concrete argument for a second stage with MLPs unfrozen.

### ⭐ The damage is NOT uniform — and NOT mathematics-specific either

The student-vs-teacher CE gap is **+0.1595 on web text (its training distribution)** and about **twice that off it** — +0.3372 on mathematics, +0.3320 on instruction-and-code — at ~19 σ, surviving a boundary-density control. **What the gap tracks is distance from the training distribution, not arithmetic.**

📌 **The numbers, the controls, the mixing arithmetic that rules out the obvious confound, and the budget argument live in `claude/first_distillkit_runs.md`** ("What stage 1 did NOT recover"). `test_r1_ce_math.py` is the measurement.

⚠️ **An earlier revision of this section read the maths result as localised damage to arithmetic.** The third-corpus control did not support that, and the claim is withdrawn. *Two corpora are a difference; three are a pattern.*

What belongs *here*, because it is about the architecture rather than about a run: **longer documents widen the gap off-distribution and leave it flat on-distribution** — the web gap held to three parts in ten thousand when mean document length doubled, while the off-distribution gap grew by a third. A fixed-size recurrent state carries a lossy summary of what came before, and the loss compounds with distance; where the student has learned the distribution well the compounding is absorbed, and where it has not it shows. ⚠️ **That is consistent with the architecture and not yet attributable to it** — any student would generalise worse off-distribution, and separating the two would need the same recipe on a non-converted student.

---

# 🔷 Track C — the GDN student ➜ **moved to `claude/building_GDN_student_track_c.md`**

*The full plan, the verified fla comparison, the geometry, the patch and the order of work now live in **`claude/building_GDN_student_track_c.md`**. This section is a one-screen summary so the three tracks still read together; do not keep two copies of the steps.*

**The research question.** Both KDA and Gated DeltaNet are gated delta-rule linear-attention layers, but they differ in exactly the place the Kimi Delta Attention paper stakes its claim: KDA's forget gate is **per channel** (`f_proj`: 2560 -> 128 -> 2560, so 2,560 decay values per token), GDN's is **per head** (`a_proj`: 2560 -> 20, twenty scalars). ⭐ **So KDA-vs-GDN here is a direct test of KDA's own central claim, in a setting its authors did not test** — distillation from a pretrained transformer rather than training from scratch. That is what makes this a study rather than a replication.

**The configuration, settled.** `expand_v=1`, `use_gate=True`, 20x128 — the drop-in. Every projection stays 2560x2560, so `transfer_attention_into_kda()` applies unchanged and the initialisation scheme is identical. **Matched on projection geometry and initialisation, not on parameter count**: 32.90 M per GDN layer against KDA's 27.61 M, and the 5.29 M difference *is* GDN's full-rank output gate against KDA's 128-dim bottleneck — one of the two architectural differences being studied.

**What it costs to build.** One new config field (`mechanism`), two GDN-only geometry extras, one adapter class, one decoder-layer branch, one line in `init_new_parameters` — and **one edit to shared code**: the padding-mask stash in `AFMKDAModel.forward` matches on a class-*name string*, so a GDN student would silently lose Track B's B7 fix. ⚠️ Re-run A3 after that one.

**What it inherits free.** The whole of Track B. `AFMKDACache`, the `get_seq_length` and `get_mask_sizes` overrides and the padding-mask stash all live in the cache and the model, not in the KDA adapter, and are mechanism-agnostic.

| variant | parameters | missing / unexpected |
|---|---|---|
| `gdn_hybrid31` ⬅ the run that matters | 5,141,828,536 | 360 / 108 |
| `gdn_flp` | 5,141,828,536 | 360 / 108 |
| `gdn_only` | 5,237,399,968 | 468 / 144 |

⚠️ **Computed, not measured** — they are what the build assertion checks.

⚠️ **Blocked on one thing:** the `VARIANTS` table and parameter assertion from `build_student.py`, so the GDN entries match that file's structure rather than being bolted on beside it.

---

## A0. The reference — what the blog post and the code each tell you

### ✅ How they initialized the weights — stated explicitly

> "Obviously for the majority of the weights they can be copied straight through from the teacher to the student. MLP, embeddings, norms, and so forth can be kept exactly the same. … **The additional parameters unique to KDA layers (like A_log, dt_bias, and so forth) are initialized from scratch.**
>
> … there isn't an equivalent of GQA for KDA. … **So I used the very elegant solution of just repeating the grouped head weights out to MHA-shaped projections.**"

⚠️ **Their conversion script was never released.** We match their *described scheme*, not their code. What makes our implementation trustworthy is different and better: it reproduces `repeat_kv`, the function the teacher's own attention calls on every forward pass. **The expansion is exact, not an approximation.**

### The three student variants, and their numbers

| Variant | Layout | Result |
|---|---|---|
| **KDA-NoPE** (hybrid) | nine blocks of "3 KDA + 1 full-attn NoPE" | best overall |
| **KDA-FLP** | first four layers full-attn; still nine full-attn total | **much worse** — unexplained |
| **KDA-Only** | no full attention | close on knowledge, drops on math; smoother long-context decay |

| | |
|---|---|
| Stage-1 convergence | token-averaged KL ≈ **0.2** on ~**300 M** tokens |
| Long-context stage | ~1 B tokens at 32k, `lr 1e-5`, **MLPs unfrozen** |
| Training throughput | **8,000 tokens/s per H100** (hybrid) |

⚠️ **The post does not give FLP's remaining five indices**, and **their training data is an Arcee-internal instruction mix**, not FineWeb-Edu.

### Architecture, settled

**Layer selection is 1-based** in the reference: `full_attn_layers: [4, 8, ..., 36]` → 0-based **3, 7, 11, 15, 19, 23, 27, 31, 35**. **No RoPE anywhere.** **`ArceeMLP` has only `up_proj`/`down_proj`** — `relu2` isn't gated. **The retained attention layers are NOT AFM's attention**: RoPE removed, plus a sigmoid output gate visible only in the code. A3b measured it: **+0.58 CE**.

### ✅ The defects — six verified

| Defect in the reference | Why it's a problem | Our status |
|---|---|---|
| Manual checkpointing loop omits `**kwargs` | Recurrent state runs across packed boundaries, silently | ✅ **A7: checkpointing irrelevant, kwargs survive** |
| No causal mask is ever built | Correct only under flash-attention | ✅ **A3: inherited `create_causal_mask`, bit-identical to teacher** |
| `**config.linear_attn_config` splatted into `KimiDeltaAttention` | `short_conv_kernel_size` isn't a parameter name → silently ignored | ✅ every argument passed by name |
| In-class `auto_map` names files that don't exist | The saved `config.json` is what matters | ✅ **A8: promoted to an instance attribute** |
| `vocab_size: 128032` vs the teacher's 128004 | Mismatched logit widths break the KL loss | ✅ inherits 128004; asserted in `test_a3.py` |
| `_tied_weights_keys` set while `tie_word_embeddings: False` | Contradictory, inert | ✅ inherited from the library |

📌 **Defect #2 is also why their model generates so easily** — never building a causal mask means there is no 4-D bias for SDPA to reject and fla gets the 2-D padding mask it wants. **Their simplicity is bought with the defect A3 caught us avoiding**, and Track B is what it cost us to be correct under every attention implementation rather than one.

### 📌 Document isolation in the reference — a difference, and then not even that

`KimiDeltaAttention` derives boundaries via `get_unpad_data` = `mask.sum(dim=-1)`. It is a *padding*-mask decoder: `nonzero()` treats every non-zero value as valid, so `1` and `2` are indistinguishable. **It cannot express a boundary inside a row.**

Their published recipe sets `packing: True` and **not** `functionary_packing`, so their stage 1 trained with cross-document context throughout.

⭐ **And so does ours.** R0 and R0b established that DistillKit's online path never delivers a segment mask, and **A9** established that the teacher could not honour one anyway. **We train in the same configuration they did** — which makes the comparison cleaner, not dirtier. Full account: `first_distillkit_runs.md` Part 4.

**Why that is not "wrong".** Packing without document isolation is ordinary practice — an accepted quality tax, not a correctness bug. Whether a fixed-size recurrent state suffers more from it than attention does is a plausible hypothesis with **no evidence behind it**.

**What *is* a genuine latent bug in the reference:** their model is **incompatible with the packing mode that would fix it**. Enable `functionary_packing` against `ArceeKDAModel` and `get_unpad_data` receives `[1×301, 2×200]`, sums it to **701**, and hands the kernel that length for a row from which `nonzero()` selected **501** tokens. *(We hit exactly this: a direct-layer probe returned `nan`.)*

---

## A1. Geometry ✅

⚠️ **AFM is native transformers, not remote code.** `trust_remote_code: true` for the *teacher* is inert.

⚠️ **Rewrite the three-dot relative imports in BOTH library copies:**

```bash
sed -i 's/^from \.\.\.\([a-z_.]*\) import/from transformers.\1 import/' modeling_arcee.py configuration_arcee.py
```

| Field | Value |
|---|---|
| `hidden_size` | 2560 |
| `num_hidden_layers` | **36** (→ 37 hidden states) |
| `num_attention_heads` / `num_key_value_heads` | **20 / 4** |
| `head_dim` | **128** |
| `intermediate_size` | 18432 |
| `hidden_act` | `relu2` |
| `vocab_size` | **128004** |
| total parameters | **4,619,184,640** |

### Parameter budget (from the **installed** fla 0.5.2)

| Component | params |
|---|---|
| q/k/v/o_proj (4 × 2560²) | 26,214,400 |
| q/k/v short convs (depthwise, k=4) | 30,720 |
| `f_proj` bottleneck | 655,360 |
| `g_proj` bottleneck (+ bias) | 657,920 |
| `b_proj`, `A_log`, `dt_bias`, `o_norm` | 53,908 |
| **KDA layer total** | **27,612,308** |
| Arcee attention layer | 15,728,640 |
| **delta per converted layer** | **+11,883,668** |
| attention gate (`gate_proj`) | +6,553,600 |

```
4,619,184,640 + 27 × 11,883,668 + 9 × 6,553,600 = 4,999,026,076   ✅ measured exactly
```

⚠️ **Honest caveat on transferring q/k.** `chunk_kda` runs with `use_qk_l2norm_in_kernel=True`: q and k are L2-normalized *inside* the kernel, so scaling `q_proj` by any constant changes nothing. **Only direction survives; magnitude is discarded.** This predicted the small measured effect in A5b.

### ⚠️ The file you downloaded is not the library you installed ✅

The `kda.py` originally placed in the project came from **GitHub's default branch**; `pip install` gives the **0.5.2 release tag**. ✅ The stale copy has been removed; **`kda_installed_0.5.2.py` is authoritative** — as is `gdn_installed_0.5.2.py` for Track C.

| | GitHub main | **Installed 0.5.2** |
|---|---|---|
| `__init__` | — | **identical**, line for line |
| unpadding | one `unpad_hidden_states(...)` helper | inline `get_unpad_data`, **overwrites `cu_seqlens`** |

**Everything the student build rests on is unaffected**, because `__init__` is identical.

---

## A2. The student class ✅

### ⚠️ transformers 5.x configs are `@strict` dataclasses

```python
@strict                                   # from huggingface_hub.dataclasses
class ArceeConfig(PreTrainedConfig):      # note: PreTrainedConfig, capital T
    model_type = "arcee"
    vocab_size: int = 32000               # ANNOTATED CLASS ATTRIBUTES
    def __post_init__(self, **kwargs):
```

**There is no `__init__`.** Subclasses declare new fields as annotated class attributes and initialize through `__post_init__`.

📌 *That `vocab_size: int = 32000` default is why both teacher and student log `bos_token_id must be … between 0 and 31999, got 128000` at load. It is dataclass-default validation, it appears identically on the unmodified teacher, and it is inherited noise — not ours.*

### ⚠️ Class attributes never reach `config.json` — the A8 bug

`save_pretrained` calls `to_dict()`, which copies `self.__dict__` — **instance attributes only**. Reading a class attribute looks identical, **which is what hides the bug**.

*The class body is a blueprint; `save_pretrained` photographs the building.*

```python
    def __post_init__(self, **kwargs):
        if self.kda_layers is None:
            self.kda_layers = []
        super().__post_init__(**kwargs)
        self.model_type = type(self).model_type      # config.json says "arcee"; ours must win
        self.auto_map = dict(type(self).auto_map)    # a class attribute is never serialised
```

### The three feature flags

| Flag | Default (= AFM) | Arcee's hybrid |
|---|---|---|
| `kda_layers` | `[]` | 27 layers, 3:1, **0-based** |
| `use_rope` | `True` | `False` (NoPE) |
| `use_attn_gate` | `False` | `True` (sigmoid output gate) |

**This is what made the verification ladder possible.** With all three at defaults the student *is* AFM bit-for-bit, so every later deviation is attributable to a flag.

### ⚠️ Do NOT override `_init_weights` — the three-attempt saga

**Attempt 1 — no `_init_weights` → new parameters got neither** checkpoint values nor a draw. Uninitialized memory, NaN in bf16.

**Attempt 2 — add it (as the reference does) → old parameters got both.** A3 went from `0.000e+00` to a logit diff of **12.0**.

**Attempt 2b — add the `_is_hf_initialized` guard.** The flag exists (291/291 parameters carry it). Still failed. Two runs of *identical code* gave `0.000e+00` and `3.404e+02` — **non-determinism is the smoking gun**.

**Attempt 3 — don't use the hook.** Initialize `missing_keys` by name after `from_pretrained` returns. **There is no code path by which a loaded weight can be touched.** ✅ A3 returned to `0.000e+00`, twice.

**Lessons:** (1) *A hook is a request to be called at a time you don't control.* (2) *Non-determinism is a diagnosis.* (3) *The reference's `_init_weights` is right for transformers 4.57 and wrong here.*

---

## Results — the whole ladder

### A3 — all-attention student == teacher ✅

| # | Check | Result |
|---|---|---|
| 1 | `missing` / `unexpected` / `mismatched` keys | all empty |
| 2 | parameters | teacher 4.6192 B, student 4.6192 B |
| 3 | max abs logit difference | **0.000e+00** (scale 22.0) |
| 4 | `len(hidden_states)` | **37 / 37**, max diff 0.000e+00 |
| 5 | layer types | 36 attention, 0 KDA |

**A3 is the regression test.** Run it after any change to the model code, *twice*. It has earned its keep four times, and **it was step B5 of Track B** — where it passed again after the cache bridge landed.

### A3b + A4 + A5 — mean CE on 8 × 2048 held-out tokens ✅

| Configuration | CE | vs teacher | Perplexity | params |
|---|---|---|---|---|
| **Teacher** | 2.3030 | — | **10.0** | 4.6192 B |
| baseline (= teacher) | 2.3030 | +0.0000 | 10.0 | 4.6192 B |
| **NoPE only** | 5.9853 | **+3.6823** | 397 | 4.6192 B |
| **attention gate only** | 2.8787 | **+0.5757** | 17.8 | 4.8551 B |
| 1 KDA layer, random | 7.2190 | +4.9160 | 1,365 | 4.6311 B |
| full hybrid, random | 8.1454 | +5.8424 | 3,452 | **4.9990 B** |

- **NoPE and the gate are isolated measurements.** RoPE has *no learned parameters*, so the whole +3.68 is attributable to lost positional information. **Thesis numbers.**
- **The damage saturates.** 27 random recurrent layers cost only +0.93 more than one — and 8.15 is far below chance (`ln(128004) = 11.76`).

### A5b — weight transfer ✅ **and confirmed end to end by R0**

⚠️ **`repeat_kv` is group-major**, so query head `h` uses KV head `h // 5`. Backwards produces a model that runs and is quietly wrong.

| Configuration | CE | Perplexity |
|---|---|---|
| 1 KDA layer, random | 6.5125 | 673 |
| 1 KDA layer, transferred | 6.4921 | 659 |
| full hybrid, random | 8.1462 | 3,455 |
| **full hybrid, transferred** | **7.9412** | **2,814** |

⭐ **R0 confirmed 7.9412 independently.** Inside DistillKit, on a different sample, through a saved-and-reloaded checkpoint, with the teacher's tokenizer and DeepSpeed in the loop, step-0 cross-entropy read **7.918**. **Two independent paths, three significant figures** — validating that the config flags built the intended architecture, that the ★ parameters were initialized without touching the loaded ones, that the group-major GQA repeat went the right way round, that all 624 tensors survived `save_pretrained`, that `auto_map` and the module cache resolved in a fresh process, that DistillKit loaded the student and not the teacher, and that the tokenizers agree. **Any one of those failing produces a model that trains without error.** This is the result to put in the thesis.

### A6 — throughput ✅ (and two corrections)

| T | KDA tok/s | attention tok/s | KDA / attn |
|---|---|---|---|
| 2,048 | 319,715 | 1,038,414 | **0.31×** |
| 16,384 | 526,868 | 533,597 | **0.99× ← crossover** |
| 32,768 | 537,073 | 314,029 | **1.71×** |

**⚠️ Correction 1.** Not "the headline figure justifying the project": KDA is 3× slower at 2048, the length stage-1 trains at, and uses **more** memory here. **The memory win is an *inference* property.**

**⚠️ Correction 2, from R0.** At the *model* level the penalty is only **~12%**, not 69% — the MLPs dilute the token-mixing cost. **A single-layer benchmark is not a model benchmark.**

📌 **B8's smoke run is consistent with this.** GSM8K generation ran at ~9 s/item for the student against ~2.3 s for the teacher — but that ratio is **confounded by output length**, because a model that fails to emit the stop string runs to the token cap. Do not quote it as a speed measurement; B9b is the measurement.

### ⚠️ The "100× smaller" headline is a PER-LAYER number — the model-level figure is much smaller for the hybrid

At 32k one attention layer's KV cache is `2 × 32768 × 4 × 128 ≈ 33.6 M` numbers; one KDA layer's recurrent state is `20 × 128 × 128 = 327,680`, **the same at 32k as at 128 tokens**. That really is ~100×, and it is the right number for one layer.

**It is the wrong number for the model**, and A6's own trap says why: *a single-layer benchmark is not a model benchmark*. The 3:1 hybrid keeps **9 of 36** attention layers, so its total inference state cannot fall below one quarter of the teacher's however good KDA is.

### ✅ B9a — inference memory, MEASURED and corrected 2026-09-20

**The check did its job.** After B4′ the KDA layers actually populate their state, and the state rides on the cache object as `AFMKDACache.fla` — so a naive walk of `past_key_values` returned **KV + KDA together**. The closed form, which predicts KV alone, therefore flagged `*** MISMATCH ***`. That is the instrument working, not a failure.

**And the mismatch carried a finding.** B6b measured **35.33 MB** for 27 KDA layers where the geometry predicted 18.06 MB:

```
measured / layer = 35.33 MB / 27 = 1,372,160 B
recurrent 20 × 128 × 128 × 4 B (FLOAT32)  = 1,310,720
conv      3 × 2560 × 4 × 2 B  (bf16, width conv_size not conv_size−1) =    61,440
                                            ----------
                                             1,372,160   ✅ exact
```

⭐ **The recurrent state is kept in float32, not bf16** — sensible, since the delta-rule state accumulates over thousands of steps and bf16 would drift — **and the conv state is `conv_size` wide, not `conv_size − 1`.** Both of my assumptions were wrong. ✅ **Confirmed by looking**: the dtype dump reads `['recurrent_state'] Tensor(1, 20, 128, 128) torch.float32 = 1280.0 KB`. The corrected constant is **1.3086 MB per KDA layer**, and both closed-form checks now pass for all three models.

#### The table — all measured

| context | teacher | **hybrid 3:1** | **kda_only** |
|---|---|---|---|
| 512 | 36.0 MB | 44.3 MB — **0.81×** | 47.1 MB — **0.76×** |
| 1,024 | 72.0 MB | 53.3 MB — 1.35× | 47.1 MB — 1.53× |
| 2,048 | 144.0 MB | 71.3 MB — 2.02× | 47.1 MB — 3.06× |
| 8,192 | 576.0 MB | 179.3 MB — 3.21× | 47.1 MB — 12.2× |
| 32,768 | 2,304.0 MB | 611.3 MB — **3.77×** | 47.1 MB — **48.9×** |

⚠️ **Revision 27 published 594 MB / 24.1 MB / 3.9× / 95.7× for this table, computed under a bf16 assumption. Those numbers are WRONG and have been deleted from this document** — if you find them quoted anywhere, they predate the measurement. **The headline is 48.9×, not 95.7×.**

⭐⭐ **The most interesting row is the first one, where the converted models use MORE memory than the teacher.** A fixed state is not a free state: it costs 1.31 MB per layer whether the context is 8 tokens or 32,768.

```
teacher KV  =  36 × 2 × T × 4 × 128 × 2 B
converted   =  n_attn × 2 × T × 4 × 128 × 2 B  +  n_kda × 1,372,160 B
→ crossover at T ≈ 670 tokens, IDENTICAL for hybrid and kda_only
```

The crossover is the same for both because it is a **per-layer** property: each converted layer trades `2 × T × 512 × 2 B` of growing KV for 1,372,160 B of fixed state, and those are equal at T = 670 regardless of how many layers you convert.

**This belongs in the thesis exactly as it is.** "Below ~670 tokens the conversion costs memory; above it the saving grows without bound, reaching 3.8× for the hybrid and 49× for KDA-only at 32k." That is a sharper and more defensible claim than a single ratio.

📌 **`peak fwd` is not the cache and must never be quoted as one.** It is 10,464 MB for the teacher and 8,160 MB for kda_only at 32k — only 22 % apart, because a full-sequence forward is dominated by activations, not by the cache. **The conversion buys almost nothing at prefill; it buys during decode.** Anyone reading a memory claim will ask this, so answer it before they do.

⚠️ **Never quote a memory ratio without saying which model it belongs to.** Per-layer, hybrid and KDA-only differ by more than an order of magnitude at 32k. ⭐ And their own numbers put **kda_only ahead of the hybrid on MMLU and ARC-C** — which, with 48.9× against 3.77×, makes `afm_kda_only_init` the most valuable second training run.

### A7 — the KDA layer honours `cu_seqlens` ✅ **bit-exact**

| Test | With `cu_seqlens` | Without | Verdict |
|---|---|---|---|
| packed vs B alone, checkpointing OFF | 3.516e-02 | 4.852e-01 | 13× reduction |
| packed vs B alone, checkpointing ON | 3.516e-02 | 4.852e-01 | **byte-identical to OFF** |
| ⭐ **B after A vs B after A′** | **0.000e+00** | 3.763e-01 | **perfect reset** |

**The residual 3.5e-2 is numerical**: the per-position profile **does not decay** (0.94 → 0.63 → 0.75 → **1.27**) where leakage would, and the values are exact binary fractions — bf16 quantization.

⚠️ **The test design mattered more than the test.** The first version flagged FAIL at a threshold I invented (1e-2), which was measuring a confound — a 501-token row chunks differently from a 200-token one. The A-vs-A′ comparison has identical shapes and arithmetic, so the expected answer is exactly zero and no threshold is needed. **When a test needs a tolerance you had to invent, look for a version that doesn't.**

### A7b ⭐ — does a `cu_seqlens` actually arrive? ✅ **fixed**

| Mask passed to the model | before | **after the fix** |
|---|---|---|
| segment-id `[1×301, 2×200]` | 3.125e+00 | **0.000e+00** ✅ |
| all-ones `[1×501]` — the control | 3.125e+00 | 3.125e+00 ✅ |

**Before the fix the two rows were identical**, not merely both large — the segment IDs had no effect whatsoever.

**Three things had to line up wrongly:** the monkey patch never injects a layer kwarg; fla 0.5.2 then derives boundaries from `attention_mask`, but `get_unpad_data` cannot split a row and computes 701 for a 501-token one; and `create_causal_mask` had already turned the mask 4-D, which our adapter drops.

```python
def build_cu_seqlens(segment_mask: torch.Tensor) -> torch.Tensor:
    row = segment_mask[0]
    lens = [n for i in range(1, int(row.max()) + 1)
            if (n := int((row == i).sum())) > 0]
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device=segment_mask.device)
    cu[1:] = torch.tensor(lens, dtype=torch.int32, device=segment_mask.device).cumsum(0)
    return cu


def is_segment_mask(attention_mask) -> bool:
    return (attention_mask is not None and attention_mask.dim() == 2
            and attention_mask.shape[0] == 1 and int(attention_mask.max()) > 1)
```

⚠️ **`shape[0] == 1` is load-bearing** — fla's varlen convention flattens the batch. ⚠️ **Pinned to fla 0.5.2.** ⭐ **And `shape[0] == 1` is exactly what B7 revisited**: batched left-padded generation arrives with `shape[0] > 1` and a genuine 2-D padding mask that KDA *does* need. `is_segment_mask` correctly returns False there, and Track B added the separate stash that routes the padding mask through.

### A7c ⭐ — and the *attention* layers? ✅ **fixed**

| Mask | first test version | with the monkey patch | **after the varlen fix** |
|---|---|---|---|
| segment-id | 1.500e+00 | 1.385e+00 | **0.000e+00** ✅ |
| all-ones control | 4.719e+00 | 5.188e+00 | 5.188e+00 ✅ |

⚠️ **Compare only *down* a column.**

**Why the monkey patch never fires — measured, not inferred:**

```
attn impl: flash_attention_2
[ATTN L3] mask: None | cu_seq_lens_q: None | cu_seqlens: tensor([0, 301, 501], dtype=torch.int32)
get_unpad_data calls: 0  []
```

The patch is correctly installed and never invoked — flash's unpad path exists to remove padding, and with none the branch is skipped. And **our `cu_seqlens` was already arriving**, under fla's name, which the flash interface doesn't recognize. **The boundaries were there; only the label was wrong.**

```python
        if is_segment_mask(attention_mask):
            cu = build_cu_seqlens(attention_mask)
            max_len = int((cu[1:] - cu[:-1]).max())
            kwargs["cu_seqlens"] = cu                # fla's spelling      -> 27 KDA layers
            kwargs["cu_seq_lens_q"] = cu             # transformers'       ->  9 attn layers
            kwargs["cu_seq_lens_k"] = cu
            kwargs["max_length_q"] = max_len
            kwargs["max_length_k"] = max_len
```

plus `kwargs.pop("cu_seqlens", None)` in `AFMKDAAttention.forward`.

#### ⚠️ The bug that hid inside the fix — a `forward` one indent too deep

Pasted with `def forward` nested **inside `__init__`**, Python accepts it: `__init__` defines a local function and discards it, so `AFMKDAModel` silently inherits `ArceeModel.forward`. A7c then read **5.188e+00 for both mask and control** — the same signature as the original A7b failure — while A3 still passed, because A3 never exercises that branch.

```python
assert "forward" in AFMKDAModel.__dict__, \
    "AFMKDAModel.forward is not defined at class level -- check indentation"
```

⭐ **Track B added a second import-time assert** beside it, on the same principle: `assert issubclass(AFMKDACache, DynamicCache)`.

### 📌 A9 — and what happened next

**A7b and A7c are correct and, in practice, inert.** R0's check 7 showed the pipeline never delivers a segment mask; R0b showed the collator sends `input_ids` and nothing else; **A9 showed the teacher cannot honour one either** — 1.302e+01 for both the segment mask and the all-ones control, because transformers discards a mask with no zeros under `flash_attention_2`.

That is what closes the question rather than deferring it. If the student isolated and the teacher did not, every token near a boundary would carry a target the student structurally cannot produce — worse than no isolation. **As things stand, student and teacher see exactly the same thing, so the distillation objective is well posed.**

**Keep the code.** A3 proves the model is bit-identical without a segment mask, and `is_segment_mask(None)` short-circuits on its first condition. Full decision record: `first_distillkit_runs.md` Part 4.

### A8 — build, save, reload ✅

```
model_type : afm_kda        class      : AFMKDAForCausalLM
kda_layers : 27             params     : 4.999026076 B
auto_map   : {'AutoConfig': 'configuration_afm_kda.AFMKDAConfig',
              'AutoModelForCausalLM': 'modeling_afm_kda.AFMKDAForCausalLM'}
```

**1. `auto_map` was a class attribute** → never serialized. Symptom: `auto_map: None`, then *"Transformers does not recognize this architecture"*.

**2. ⚠️ Transformers copies only ONE level of relative imports for remote code.**

```
5401da…/  (AutoConfig entry)    configuration_afm_kda.py, configuration_arcee.py        ✓
afac30…/  (AutoModel entry)     modeling_afm_kda.py, configuration_afm_kda.py,
                                modeling_arcee.py … configuration_arcee.py MISSING      ✗
```

**Fix: make every file in the closure a first-level import of the entry module.**

```python
from .configuration_arcee import ArceeConfig  # noqa: F401
```

### The build procedure

```bash
sbatch --export=ALL,VARIANT=hybrid31 build_student.sbatch   # ~20 GB RAM: a GPU node
# runs:  python afm_kda/build_student.py --variant $VARIANT [--force] [--no-transfer]
```

**Why the `.py` copies:** a HuggingFace model is a *directory* — `config.json` + safetensors + the architecture files. `from_pretrained` reads `config.json`, sees `auto_map`, and imports those files **from that same directory**.

---

## Evaluating the student — the protocol, and the traps

The frozen protocol lives in `claude/eval_zeroshot_afm.sbatch`, `claude/eval_mmlu_afm.sbatch` and `claude/eval_generative_afm.sbatch`:

| tasks | shots | metric |
|---|---|---|
| ARC-Challenge, HellaSwag, WinoGrande | **0** | `acc_norm`, `acc_norm`, `acc` |
| MMLU | **5** | `acc` |
| **GSM8K** | **5** *(lm-eval's task default)* | `exact_match` |
| **IFEval** | **0** *(its only mode)* | — |

matching RADLADS and the practical work. **No chat template** — every model here is a base model. **Do not change a shot count once a student has been measured.**

⚠️ **GSM8K at 0-shot would measure nothing** — a base model writes prose instead of an answer and both teacher and student score ~0. 5-shot is the setting where the number is interpretable.

⚠️ **Pass `use_cache=True` explicitly in `model_args`, for every model.** The Trainer writes `use_cache: false` into every checkpoint it saves, and Arcee ship their published configs the same way — so a flag flipped on one side only makes the student O(n) and the teacher O(n²). State the setting once in the methods section; identical for both is the rule.

⚠️ **`--limit N` is a pipeline test, not a measurement.** 20 items gives ±0.11 on a mid-range score, and 0/20 is consistent with a true rate of up to ~14 %. Smoke on the debug queue, then run the full set before quoting anything.

### ⚠️ Offline, a hub repo id is not a path — use the snapshot directory

`MODEL_PATH=arcee-ai/AFM-4.5B-Base` failed with `OSError: We couldn't connect to 'https://huggingface.co' … and couldn't find them in the cached files`, even though the teacher was plainly in the cache.

`HF_HOME=$WORK_LUKAS/hf_cache` puts the hub store at `$WORK_LUKAS/hf_cache/**hub**/`, but the eval scripts also set `HUGGINGFACE_HUB_CACHE=$WORK_LUKAS/hf_cache`, which **overrides** that and points the resolver one directory too high. It looks for `hf_cache/models--arcee-ai--AFM-4.5B-Base`, finds nothing, tries the network, and is offline.

⭐ **Prefer the explicit snapshot path, and not only because it works.** The snapshot hash *is* a provenance record: a repo id resolves to whatever revision the cache happens to hold, while `dab7922f…` pins the exact bytes that produced the number in the thesis.

⚠️ **`eval-env2` needs `flash-linear-attention`** — installed 2026-09-19. Without it the student's remote code raises `ImportError` at load. Every eval sbatch now prints the installed version in its provenance block; read that line before assuming a crash is our code.

⭐ **The 5.8.1 / 5.9.0 environment skew is closed by measurement.** `test_r1_ce.py` returns identical numbers in both, and `DynamicCache.get_mask_sizes` is byte-identical between the two versions. Record both versions with the numbers and stop worrying about it.

---

## Next

Track A is closed. Track B is closed. **R1 is trained and benchmarked.**

| | |
|---|---|
| ✅ **MMLU** — teacher **0.6500**, student **0.3055** | **13.9 % relative.** It fell hardest, as predicted, and **past the 55–65 % band this project pre-registered**. Full analysis: `first_distillkit_runs.md` |
| 🔶 **B8** — GSM8K ✅ full (0.0243 strict vs 0.5353) · IFEval ⬜ | and GSM8K is **not zero**: 5.8 σ, ~32 of 1,319 |
| ⬜ **B9b** — decode throughput vs context length | the speed half of the architectural claim |
| ⬜ **Rebuild the four students** from `d7cd0a0` | hash the weights first |
| ⬜ **Track C** — the GDN student | unblocked |
| 🔶 **Budget decision** with Anamaria | see `first_distillkit_runs.md`, "Open decisions" |

### Scope decision: the long-context stage

> "I collapsed the first two distillation stages into a single one with frozen MLP parameters … **then allowed the long context fine tune to pick up any slack necessary in adjusting the MLP layers.**"

RADLADS's pipeline is three stages; Arcee merged 1 and 2 and used stage 3 as the place where MLPs unfreeze. **Drop it entirely and nothing outside the attention layers ever adapts** — and RADLADS reports that freezing MLPs and embeddings hurts.

| Option | What | Cost |
|---|---|---|
| **(a) Stage 1 only** | Evaluate at ≤2048, state the limitation | cheapest |
| **(b) Stage 1 + a short full-parameter stage at 2048** | Unfreezes MLPs; **RADLADS stage 2** | ~1 extra stage-1-sized run |
| **(c) All three stages** | What Arcee did | expensive |

**Recommended: (b).** ⭐ **And the GSM8K result is now the concrete argument for it** — not "more tokens would be nice" but "one benchmark reads zero for every student and therefore cannot discriminate between them."

---

## Checklist

```
--- TRACK A: COMPLETE ---
[x] A0. reference code AND blog post read; conversion script is NOT public
[x] A0. document isolation in the reference characterized -- and we match it
[x] A1. geometry + parameter budget; predicted 4,999,026,076, MEASURED the same
[x] A1. facts re-verified against the INSTALLED fla 0.5.2
[x] A2. @strict dataclass style; model_type AND auto_map promoted to instances
[x] A2. new parameters via explicit init_new_parameters(), NOT _init_weights
[x] A3. five checks pass; logit diff 0.000e+00; REPRODUCIBLE across two runs
[x] A3b. NoPE +3.6823 CE, attention gate +0.5757 CE (isolated)
[x] A4/A5. hybrid runs fwd+bwd; grads 624/624; 4.9990 B
[x] A5b. transfer verified (20/20 heads) and measured (-0.21 CE); CE 7.9412
[x] A5b. CONFIRMED END TO END: R0 measured 7.918 inside DistillKit
[x] A6. KDA vs attention curve; corrected twice (headline, then 0.31x -> 0.88x)
[x] A7 / A7b / A7c. cu_seqlens honoured, arrives, and reaches both layer types
[x] A9. the TEACHER cannot honour a segment mask -- question closed
[x] A8. student saved AND reloaded in a fresh process as AFMKDAForCausalLM
[x] afm_kda committed at 3aafffc; four students built from it
[x] build_info.json verified: 441/108, 4,999,026,076 params, fla 0.5.2

--- TRACK B: GENERATION -- COMPLETE.  Full record in claude/track_b.md ---
[x] B0 / B0b. fla-Cache-everywhere FAILS on transformers 5.x; cause dropped
[x] B9a. memory curve MEASURED: 2304 / 611 (3.77x) / 47.1 MB (48.9x) at 32k
[x] B2'-B4'. the two-cache bridge applied
[x] B5. A3 twice -> still 0.000e+00
[x] B1+B6/B6b/B6e. cached == uncached; teacher control settles the magnitude
[x] B7 / B7b. batched + left-padded; get_mask_sizes and the padding-mask stash
[x] B7c. one real generate() call -- fluent prose, most-padded row cleanest
[x] afm_kda committed at d7cd0a0, clean tree; docs/ versioned in the repo
[x] B8. GSM8K FULL (1,319): student 0.0243 strict vs teacher 0.5353
[ ] B8. IFEval, FULL run -- the last outstanding benchmark
[ ] B9b. decode throughput vs context length

--- TRACK C: GDN (designed, not built -- UNBLOCKED) ---
[x] gdn_installed_0.5.2.py saved as the authoritative source
[x] parameter breakdown done; the gap is expand_v + the output-gate rank
[x] framing settled: this tests KDA's fine-grained-gating claim
[x] configuration chosen: expand_v=1, use_gate=True, 20x128 (the drop-in)
[ ] check GDN's unpad/repad block against KDA's -- do NOT assume B7 transfers
[ ] AFMGDNLinearAttention adapter + `mechanism` config field
[ ] build_student.py VARIANTS entry + parameter assertion + a finalize marker
[ ] build afm_gdn_hybrid31_init

--- OPEN, NOT BLOCKING ---
[x] MMLU, teacher and student -- 0.6500 / 0.3055 = 13.9% relative
[ ] ARC-C at 5-shot -- DIAGNOSTIC: does a held benchmark collapse when the
        prompt becomes a 5-shot template?  Separates ICL from format distance
[ ] rebuild the four students from d7cd0a0 (hash the weights first)
[ ] flp layout CONFIRMED against AFM-4.5B-Base-KDA-FLP/config.json (currently OURS)
[ ] delete the leftover `_is_hf_initialized` diagnostic from test_a3.py
[ ] remove the now-dead .contiguous() line from AFMKDAAttention.forward
[x] flash-linear-attention 0.5.2 + fla-core 0.5.2 installed into eval-env2
[x] the 5.8.1 / 5.9.0 env skew: CLOSED by measurement, twice
```

## Traps, collected

- **⚠️ Read how a configuration flag is consumed before building on what its name implies.** `functionary_packing` reads as "enable packing"; it means "assume the data is already packed in functionary's format".
- **⭐ Read the library before patching around it.** transformers 5.8.1 already implements the hybrid `get_mask_sizes` redirect; it just could not see our layers. Four greps replaced a week of guessing.
- **⚠️ Never quote a memory ratio without saying which model it belongs to.** Per layer KDA is ~100x smaller at 32k; the 3:1 hybrid is 3.77x because it keeps 9 attention layers; kda_only is 48.9x.
- **⚠️ Delete a superseded table, do not leave it below the corrected one.** Revision 27's 95.7x sat under revision 28's 48.9x in this document for two days, both looking authoritative.
- **⚠️ A pass under a configuration that bypasses the patched code is not evidence about the patch.** `flash_attention_2` never builds a 4-D mask, so B7's clean fa2 rows said nothing about the padding fix.
- **⚠️ A broadcastable wrong shape does not raise.** A mask of width 1 against 301 keys broadcasts silently; the error surfaces three layers down as a contiguity complaint about a stride the kernel created itself.
- **⚠️ Preventing a crash is not fixing a bug.** Dropping the 4-D mask kept fla's assertion quiet and let pad tokens into the recurrent state instead.
- **⚠️ A hand-rolled loop is not the code path the harness uses.** Five Track B tests validated a decode loop we wrote; none called `generate()`.
- **⚠️ A setting you fixed on one model is a setting you have now made asymmetric.** Flipping `use_cache` on the student silently created a confound with the teacher.
- **⚠️ `--limit N` is a pipeline test, not a measurement.** 0/20 is consistent with a true rate up to ~14%.
- **⚠️ Separate what you observed from what you inferred.** B0 observed three identical failures; the *reason* was a hypothesis.
- **⚠️ A ten-minute probe can retire a day of work.** B0 cost one debug-queue job and killed an approach that would have taken three steps to discover was impossible.
- **⚠️ The remote-code module cache is under `$HF_HOME`, not `~/.cache/huggingface`.** Clearing the wrong one is silent.
- **⚠️ `save_pretrained` DOES copy remote-code `.py` files into a checkpoint.** Look in the directory before writing a workaround for a problem that isn't there.
- **⚠️ Fix the chain, not the artefacts.** Copying the model file into every checkpoint is correctness by remembering; committing the repo and rebuilding the init students is correctness by construction.
- **⚠️ A signal that always fires is no signal.** An un-ignored `logs/` directory makes the dirty-tree provenance check permanently red.
- **⚠️ The reference targets transformers 4.57 and we run 5.x.** That gap has now changed the answer seven times — `_init_weights`, the causal mask, checkpointing kwargs, `auto_map`, `@strict` configs, the cache design, and `get_mask_sizes`.
- **⚠️ Check what a guard sees on every path that reaches it, including evaluation.** A `use_cache` check written for training would have fired on R1's first eval pass and killed a four-hour run.
- **⚠️ The slow, obviously-correct workaround is usually the oracle for testing the fast fix.** `use_cache=False` generation was too slow to ship and exactly right as B1's ground truth — and, later, as the control that exonerated the cache on GSM8K.
- **⚠️ A capability you cannot exercise is a claim you cannot make.**
- **A test that supplies the input it assumes confirms the assumption, not the system.** A7 → A7b → A7c → R0 → R0b → A9.
- **⚠️ A probe that can only report success tells you nothing when it fails.** Instrument the decision, not the happy path.
- **⚠️ stdout and stderr are two different files under SLURM.** `traceback.print_exc(file=sys.stdout)`.
- **⚠️ `traceback.print_exc(limit=N)` truncates from the OUTSIDE.** It hid the raising frame twice on Track B.
- **⚠️ A discriminating test needs an input the two branches can disagree on.** Sixteen arbitrary token ids hid a real bug behind a 4% difference.
- **`X.__dict__` answers "defined here", not "defined by this library".**
- **⚠️ A cache can lie about its length.** `get_seq_length()` reads layer 0, and in a hybrid layer 0 may store nothing.
- **Count the calls before trusting a patch.** *A patch that is installed is not a patch that runs.*
- **⚠️ A `def` one indent too deep is a silent no-op.** Assert `"forward" in Cls.__dict__` at import.
- **Two identical numbers are a finding**, stronger than a merely-large number — A7b, the vanished `forward`, A9, and the byte-identical cached/uncached GSM8K output.
- **Always include a negative control.**
- **The same value can travel under two names.** The A7c probe showed `cu_seqlens` already arriving, unrecognized.
- **⚠️ `HUGGINGFACE_HUB_CACHE` overrides `HF_HOME`'s `hub/` subdirectory.** **Prefer an explicit snapshot path — it is also a provenance record.**
- **Do not assume one library layer's quirks carry to its sibling.** KDA 0.5.2 overwrites a supplied `cu_seqlens`; GDN 0.5.2, same release, does not.
- **Make the build record a number you can check by hand.** `loading_info` 441/108 re-verifies the whole inventory for free on every rebuild.
- **`dict.get(key, default)` turns a wrong key into a plausible answer.**
- **A library file you downloaded is not the library you installed.**
- **For a remote-code model, `config.json` is the contract; the class is decoration.**
- **HF copies only ONE level of relative imports.** Clear the module cache after any `.py` change *or rename*.
- **A variable assigned inside an sbatch script beats `--export`.** Use `${VAR:-default}`.
- **A dirty tree makes a recorded hash a half-truth.**
- **Never override `_init_weights` on a subclass of a library model in transformers 5.**
- **Run any suspect test twice.**
- **If a test needs a tolerance you invented, redesign the test.** Three invented tolerances broke Track B before the kernel source was read.
- **⚠️ An absolute floating-point tolerance is meaningless without a magnitude.** One bf16 ulp at logit scale 34.5 is 0.25.
- **A NaN can be the answer, not a broken test.**
- **GQA expansion is group-major.** Query head `h` uses KV head `h // n_rep`.
- **transformers 5.x configs are `@strict` dataclasses.**
- **`_can_record_outputs` must name your subclassed layer**, or `hs_cosine` silently compares nothing.
- **`re.search` matches anywhere; `re.match` only at position 0.**
- **`use_qk_l2norm_in_kernel=True` discards transferred q/k magnitude.**
- **The reference's layer indices are 1-based; ours are 0-based.**
- **Never splat a config dict into a fla layer constructor.**
- **RoPE has no learned parameters.** **36 layers, 37 hidden states.**
- **A single-layer benchmark is not a model benchmark.** A6's 0.31× became ~0.88× at the model level — and a generation-speed ratio is confounded by output length.
- **KDA's memory advantage is an inference property**, not a training one.
- **A checkpoint's directory name is a label; `config.json` is the definition.**
- **Two independent measurements agreeing is worth more than either alone.** A5b's 7.9412 and R0's 7.918.
- **⚠️ A modest perplexity penalty is not a modest benchmark penalty.** +0.16 nats (17% perplexity) coexists with 79–89% relative retention on three 0-shot benchmarks and **4–14% on the two 5-shot ones**.
- **⭐⭐ A second data point can falsify the explanation you built on the first.** "Errors compose multiplicatively along a generated chain" accounted for GSM8K perfectly and is powerless against MMLU — one forward pass, an argmax over four options, and it collapsed anyway. The property the two share is the **shot count**, which nobody was looking at.
- **⚠️ Pre-register the band, then honour it when you miss.** This document wrote "55–65 % relative on MMLU would be the honest, interesting result". It came in at **13.9 %**. A pre-registration is worth nothing unless a miss is recorded as a miss.

## Sources

- `distillkit_blogpost_kda.txt` — [Distilling Kimi Delta Attention into AFM-4.5B](https://www.arcee.ai/blog/distilling-kimi-delta-attention-into-afm-4-5b-and-the-tool-we-used-to-do-it), Charles Goddard
- `RADLADS.pdf` — three-stage pipeline; de-novo-initialization and frozen-MLP ablations; **the 0-shot / 5-shot-MMLU evaluation convention**
- `kda_installed_0.5.2.py` and `gdn_installed_0.5.2.py` — **the authoritative fla sources**
- `modeling_arcee_kda.py` — the reference implementation; ⭐ read for what it *avoids*
- transformers 5.8.1 / 5.9.0, read directly: `cache_utils.py`, `masking_utils.py`, `integrations/sdpa_attention.py`, `models/qwen3_next/`
- `main.py`, `trainer.py`, `signals.py`, `monkey_patch_packing.py`, `pack_logits.py` (DistillKit)
- `claude/track_b.md` — the generation track, complete
- `claude/eval_zeroshot_afm.sbatch`, `claude/eval_mmlu_afm.sbatch`, `claude/eval_generative_afm.sbatch` — the frozen evaluation protocol
- [arcee-ai/AFM-4.5B-Base](https://huggingface.co/arcee-ai/AFM-4.5B-Base) · [AFM-4.5B-Base-KDA-Only](https://huggingface.co/arcee-ai/AFM-4.5B-Base-KDA-Only) — the latter's config carries `use_cache: false`