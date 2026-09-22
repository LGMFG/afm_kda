# Track A — Building the KDA Student ✅ **COMPLETE**

*Companion to `claude/startup_plan_and_pitfalls.md` (step 7 there points here) and `claude/first_distillkit_runs.md` (**the live document**). This one covers the model itself.*

*Revision 30 (2026-09-20) — ⚠️ **the memory numbers are corrected by measurement.** Once B4' made the KDA layers actually store state, B6b measured **35.33 MB** for 27 layers where I had computed 18.06: the recurrent state is kept in **float32**, and the conv state is `conv_size` wide. The per-layer constant is **1.3086 MB**, so at 32k the hybrid is **3.77x** and kda_only **48.9x**, not 3.9x and 95.7x. ⭐ And the fixed state is not free: below a crossover at **T = 670 tokens** a converted model carries MORE state than the teacher.*

*Revision 29 (2026-09-20) — **Track B moves out of this document** into `claude/track_b.md`, which now holds the whole plan, the patch and the scripts in one place; what remains here is a one-screen summary. This document is back to what it is for: **how the student was built (Track A, closed)** and **the GDN sibling (Track C, designed)**.*

*Revision 28 (2026-09-20) — **B9a measured and B0b closed.** The memory curve is real: teacher 2,304 MB at 32k, hybrid **594 MB (3.9×)**, kda_only **24.1 MB (95.7×)**, with the KDA state flat at 18.06 MB from T=512 to T=32,768 — all three closed-form checks OK. ⚠️ The KDA column is *computed*, not measured, until B4′. B0b was read and is **inconclusive**; the question is dropped because B2′ makes it unreachable. **Track B is committed to B2′–B4′**, whose patch is now written out in full.*

*Revision 27 (2026-09-20) — **corrects the memory headline and unblocks half of B9.** The "100× smaller state" is a **per-layer** number; at the model level the 3:1 hybrid is capped at **~3.9×** at 32k because it keeps nine attention layers, while **kda_only reaches ~96×**. That makes hybrid-vs-KDA-only the project's central quality/memory trade-off. And the memory curve is **measurable today** — a cache is built by any forward with `use_cache=True` — so B9a needs no bridge. Track B itself is paused at **B0b**: read the thirty lines of transformers that actually raise, before committing to B2′.*

*Revision 26 (2026-09-20) — **B0 is closed and the plan changed because of it.** The full traceback shows the failure is inside transformers' own model wrapper, **before any of our code**, triggered purely by the cache object's type — so converting to fla's `Cache` at the top is not viable on fla 0.5.2 + transformers 5.x. **Track B switches to the two-cache design** (B2′–B4′ below), which leaves `ArceeModel.forward` untouched — the thing A3 depends on. Also: both `config.use_cache` and `generation_config.use_cache` are `False`, so **the R1 checkpoint already generates correctly today, just slowly.***

*Revision 25 (2026-09-19) — **B0 ran.** fla's `Cache` *is* a transformers `Cache` subclass, so the bridge is viable; the forward still raised, and the evidence says it raised in **our** attention layer (B3's job), not in the inherited model. Three corrections to revision 24 recorded below, including one to a claim I had asserted twice: **the student's `config.use_cache` is `False`, not `True`.** ⚠️ The probe's own tracebacks went to stderr and were lost from the `.out` file — fixed, and recorded as a trap.*

*Revision 24 — **the generation gap is promoted to Track B and given a stepped plan (B0–B8)**, in the same shape as Track A's ladder, with the argument for why it is not optional: without a working recurrent cache the thesis's central architectural claim — KDA's ~100× smaller inference state — cannot be measured at all. The GDN student becomes **Track C**. Also records the offline `MODEL_PATH` trap found while running the teacher baseline.*

*Revision 23 — added the GDN parameter breakdown and the first sketch of the cache bridge. Revision 22 — closed against the real pipeline: R0's 7.918 met A5b's 7.9412; A9 showed the teacher cannot honour a segment mask. Revision 21 — `build_info.json` verified. Revision 20 — committed at `3aafffc`; four students rebuilt. Revision 19 — A7c closed. Revision 17 — corrected rev 16 on fla 0.5.2. Revision 16 — A7b added.*

---

## 🗺 The three tracks

| Track | What it is | State |
|---|---|---|
| **A** | Build the KDA student and prove it is the model we think it is | ✅ **complete** — `3aafffc`, four students, R1 training |
| **B** | Give it a working recurrent cache so it can **generate** | ⬜ **patch written** — full plan in `claude/track_b.md` |
| **C** | Build the **GDN** sibling student for the mechanism comparison | ⬜ designed, not built |

Track B gates GSM8K, IFEval, and every inference-side measurement. Track C gates the comparison that makes the thesis a study rather than a replication. **Neither blocks R1.**

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
| **generation / KV cache** | ✅ **works** — they unified on fla's `Cache` | ⬜ **not built** — **Track B** |
| lines we own | ~500 | ~200 |
| ablations | edit the file | one config flag |

⚠️ **Revision 16 got the KDA row backwards** and was corrected in rev 17. Neither implementation got document isolation for free.

**The honest scorecard:** our choice bought training correctness and a fifth of the code; theirs bought a complete model including generation. **Neither is wrong — they optimise for different things, and ours optimised for the right thing first.** Track B is the bill for that ordering, and it comes with their file as a worked answer.

---

## Where the student lives

| Thing | What it is | Where |
|---|---|---|
| **DistillKit** (your fork) | the *trainer* | `$WORK_LUKAS/distillkit` |
| **AFM-4.5B-Base** | the *teacher*, unmodified | `$HF_HOME/hub/models--arcee-ai--AFM-4.5B-Base` |
| **Student source** | the model *definition* + tests | `$WORK_LUKAS/afm_kda` (git, **`3aafffc`**) |
| **Student checkpoint** ⬅ what the YAML points at | built, saved, reload-verified | **`$WORK_LUKAS/students/afm_kda_hybrid31_init`** |
| *(reference)* | Arcee's converted student — read, don't copy | `$WORK_LUKAS/reference_kda_nope` |

```
afm_kda/                       (git repo -- code only)
  __init__.py                  (empty -- makes the relative imports resolve)
  configuration_arcee.py       library copy, imports absolutised   } baseline
  modeling_arcee.py            library copy, imports absolutised   }  commit
  TRANSFORMERS_VERSION
  configuration_afm_kda.py     yours
  modeling_afm_kda.py          yours
  build_student.py             builds + saves a student  (--variant)
  test_a3.py  / .sbatch        A3 -- the regression test
  test_a4.py  / .sbatch        A3b + A4 + A5
  test_a5b.py / .sbatch        weight transfer: verification + ablation
  test_a6_curve.py / .sbatch   KDA vs attention across sequence lengths
  test_a7.py  / .sbatch        the layer honours cu_seqlens
  test_a7b.py                  cu_seqlens ARRIVES -- the KDA runtime path
  test_a7c.py                  the same question for the ATTENTION layers
  test_a7c_probe.py            counts _get_unpad_data calls
  test_a9.py                   can the TEACHER honour a segment mask?  (no)
  .gitignore                   __pycache__, *.safetensors
```

⚠️ **Keep the checkpoints OUT of the git repo.** ~10 GB of safetensors inside `afm_kda/` would make `git add -A` unusable — hence the separate `students/` directory.

⚠️ **Editing `afm_kda/` does not change a checkpoint.** Each `students/*_init/` directory holds its own copies of the five `.py` files, and `from_pretrained` imports *those*. So you can edit the repo freely while a job is queued or running — **but the `cp` re-sync loop below is what makes a change take effect**, and a queued job will pick up whatever is in the checkpoint when it starts.

### The four students ✅ *(all built from `3aafffc`, 2026-09-18)*

| Variant | Directory | What defines it |
|---|---|---|
| **hybrid 3:1 — the main run** | `students/afm_kda_hybrid31_init` | 27 KDA layers, `use_rope: false`, `use_attn_gate: true` |
| KDA-only | `students/afm_kda_only_init` | all 36 layers KDA — **the clean test bed for A7b** |
| FLP | `students/afm_kda_flp_init` | full attention at 0,1,2,3 + 5 more — ⚠️ **layout not confirmed against theirs** |
| de-novo ablation | `students/afm_kda_hybrid31_denovo_init` | same as hybrid31, `--no-transfer` (RADLADS's ablation) |
| *(Track C, planned)* | `students/afm_gdn_hybrid31_init` | Gated DeltaNet — see Track C below |

Convention: `afm_<mechanism>_<layout>_<stage>` — `init` before training, the run name after.

⚠️ **The directory name is only a label.** What defines the model is `config.json`.

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

⚠️ **These four builds record `"dirty": true`.** The five `code_files` copied into each checkpoint are the committed ones, so the models are honest, but a dirty flag means the hash alone does not reconstruct the tree. **Commit before every build and run.**

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

⚠️ **Do not run this while a job is queued or running** unless you intend that job to use the new code. **This is the single rule that makes Track B safe to develop while R1 trains.**

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

### The 9 attention layers — 0-based indices 3, 7, 11, 15, 19, 23, 27, 31, 35

```
self_attn  (AFMKDAAttention)
├── q_proj      [2560 × 2560]   ← teacher, direct
├── k_proj      [ 512 × 2560]   ← teacher, direct   (narrow: GQA, 4 KV heads)
├── v_proj      [ 512 × 2560]   ← teacher, direct
├── o_proj      [2560 × 2560]   ← teacher, direct
└── gate_proj   [2560 × 2560]   ★ NEW, random       (only if use_attn_gate)
```

📌 **These nine layers are where Track B does most of its work.** They are the only ones that currently touch `past_key_values`, and they touch it with **transformers' API**, which is the API the bridge has to replace.

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

⚠️ **Row three is how A7b's bug happened**, and **row four is Track B in one line.**

### The proportions

| | parameters | share |
|---|---|---|
| From the teacher | ~4.90 B | **98%** |
| New, random | ~97 M | **2%** |

**Only 2% of the student is new** — but that 2% is the entire mechanism by which tokens exchange information, which is why the model scores CE 7.94 instead of 2.30.

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

# 🔶 Track B — Generation ➜ **moved to `claude/track_b.md`**

*The full plan, the patch, the scripts and the status board now live in **`claude/track_b.md`**. This section is a one-screen summary so the three tracks still read together; do not keep two copies of the steps.*

**The problem.** The student can score text but not write it, and the failure is silent. Generation keeps a cache; the nine attention layers keep a KV cache, the twenty-seven KDA layers should keep a fixed 20 × 128 × 128 recurrent state, and our adapter currently hardcodes `past_key_values=None, use_cache=False` so they keep nothing. Continue from a cache and the attention layers are right while the KDA layers behave as if the text just started — fluent, wrong, no error.

**What it blocks:** GSM8K, IFEval, and decode-throughput-vs-context-length. **What it does not block:** the four likelihood-scored benchmarks, and — established 2026-09-20 — the inference **memory** curve, which any forward with `use_cache=True` already builds.

**Where it stands.**

| | |
|---|---|
| B0 / B0b | ✅ closed. Arcee's "fla `Cache` everywhere" design **fails on transformers 5.x**; the cause is unresolved and deliberately dropped |
| B9a | ✅ **measured** — teacher 2,304 MB at 32k, hybrid 594 MB (3.9×), kda_only 24.1 MB (95.7×), KDA state flat |
| B2′–B4′ | ⬜ **patch written** — `AFMKDACache(DynamicCache)` carrying an fla cache as an attribute |
| B5, B1+B6, B7, B8, B9b | ⬜ scripts and order in `track_b.md` |

⭐ **A reprieve worth knowing:** `config.use_cache` and `generation_config.use_cache` are both `False`, so `generate()` already recomputes the prefix every step — correct, just O(n²). GSM8K is runnable today; IFEval is not. **Do not set `use_cache: true` before the bridge lands.**

---

# 🔷 Track C — the GDN student (designed, not yet built)

*`gdn_installed_0.5.2.py` in the project folder is the authoritative source — `fla/layers/gated_deltanet.py` from the installed 0.5.2, not GitHub.*

## ⭐ Why this is the right second mechanism

Both KDA and Gated DeltaNet are gated delta-rule linear-attention layers, so at first glance the comparison looks narrow. **It isn't.** Look at how each one computes its forget gate:

| | forget gate | granularity |
|---|---|---|
| **KDA** | `f_proj`: 2560 → 128 → 2560 | **per channel** — 2560 decay values per token |
| **GDN** | `a_proj`: 2560 → num_v_heads | **per head** — 20 scalars per token |

**Fine-grained channel-wise gating is precisely what the Kimi Delta Attention paper claims as its advance over Gated DeltaNet.** So a KDA-vs-GDN comparison here is a direct test of that claim **in a setting its authors did not test** — distillation from a pretrained transformer rather than training from scratch. That is a research question, not a horse race, and it is what makes GDN the right choice rather than an arbitrary one.

## Why GDN costs more at the same hidden size

Both layers take 2560 in and give 2560 out. The difference is entirely internal.

| component | **KDA** (20×128, expand_v=1) | **GDN** (recommended: 1920 key, expand_v=2) |
|---|---|---|
| q_proj | 2560→2560 — 6.55 M | 2560→1920 — 4.92 M |
| k_proj | 2560→2560 — 6.55 M | 2560→1920 — 4.92 M |
| v_proj | 2560→2560 — 6.55 M | 2560→**3840** — 9.83 M |
| o_proj | 2560→2560 — 6.55 M | **3840**→2560 — 9.83 M |
| **output gate** | 2560→**128**→2560 — 0.66 M | 2560→**3840** — 9.83 M |
| **forget gate** | 2560→**128**→2560 — 0.66 M | 2560→20 — 0.05 M |
| misc | 0.09 M | 0.05 M |
| **total** | **27.6 M** | **39.4 M** |

**Reason one — `expand_v=2`.** GDN widens the value stream to twice the key width, inherited from Mamba2: the recurrent state stores value vectors, and wider values mean more memory capacity. KDA uses `expand_v=1`. Setting GDN to `expand_v=1` removes this difference entirely.

**Reason two — the output gate is full-rank in GDN, low-rank in KDA.** 9.83 M against 0.66 M, **fifteen times**. Even at `expand_v=1` it is 2560→2560 = 6.55 M, still ten times KDA's. This one cannot be tuned away; it is an architectural choice.

📌 **GDN's recommended numbers are themselves a budget-matching convention, not a performance prescription.** The docstring is explicit: `0.75 × hidden` and `expand_v=2` exist so that a GDN layer costs `6 × hidden²`, *matching Mamba2*. Deviating from them to match KDA's budget instead is the same kind of move, anchored to a different reference. It is ordinary practice, not a violation.

## The three candidate configurations

| Config | Per layer | Student | A5b transfer | Character |
|---|---|---|---|---|
| **`expand_v=1`, `use_gate=True`, 20×128** ⬅ **recommended first** | ~32.8 M | ~5.13 B | ✅ **works unchanged** | drop-in; 2.6% heavier than the KDA student |
| `expand_v=2`, 0.75h recommendation | ~39.4 M | ~5.32 B | ❌ shapes don't fit | GDN as its authors intend it |
| `expand_v=1`, `use_gate=False`, 20×128 | ~26.3 M | ~4.98 B | ✅ works | closest parameter match, but removes GDN's gate |

**Take the drop-in first.** Not because it is parameter-matched — it isn't quite, 32.8 against 27.6 in the mixing layers — but because it keeps *everything else identical*: same projection shapes (`q/k/v/o` all 2560×2560), same weight transfer, same initialization scheme, same adapter pattern. **The initialization difference would have been a far bigger confound than 19% more mixing parameters**, and the drop-in eliminates it.

⭐ **And the residual gap is itself the finding.** After matching geometry, the 5.2 M/layer difference is *exactly* GDN's full-width output gate versus KDA's bottleneck. That is not an artefact of the config choice — it is one of the two architectural differences being studied. Report both parameter counts and say you matched projection geometry and initialization rather than parameter count.

The third row is interesting later: simultaneously the closest parameter match **and** a gating ablation. Not the first experiment.

## What building it needs

Small, and mostly already in place:

1. **`configuration_afm_kda.py`** — add a `mechanism: str = "kda"` field alongside `kda_layers`, and a matching set of `gdn_*` geometry fields.
2. **`modeling_afm_kda.py`** — a sibling `AFMGDNLinearAttention` adapter. Same contract as `AFMKDALinearAttention`: absorb `position_embeddings`, drop non-2-D masks, return two values, pass `cu_seqlens` through. **If Track B is done first, it must also speak fla's `Cache`** — which is free, since GDN handles it natively the same way KDA does.
3. **`build_student.py`** — one more entry in the `VARIANTS` table. `transfer_attention_into_kda()` works unchanged under the drop-in config, since the projection shapes match.
4. A fresh **parameter-count assertion** for the new geometry, in the same style as `KDA_DELTA = 11_883_668`.
5. **A3 equivalent:** GDN has no "flags off = the teacher" property, so the regression test is the parameter count plus forward/backward plus a CE measurement against the KDA student at initialization.

⚠️ **A useful asymmetry spotted while reading.** GDN 0.5.2 has the guard KDA lacks:

```python
cu_seqlens = kwargs.get('cu_seqlens')
if cu_seqlens is None and attention_mask is not None:      # KDA has no `cu_seqlens is None`
```

so GDN does **not** overwrite a supplied `cu_seqlens`. Two layers, same library version, opposite behaviour. Moot for us — nothing delivers a segment mask — but do not assume one layer's quirks carry to another. *(GDN's closing `pad_input` references `indices`, which is only bound inside that branch, so supplying both a `cu_seqlens` and a mask would raise `NameError` — the mirror image of KDA's silent overwrite.)*

⚠️ **Do not write this code while an R1-class job is queued**, and do not interleave it with Track B — both edit `modeling_afm_kda.py`, and a bug in one would be attributed to the other.

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
| Long-context stage | ~1 B tokens at 32k, `lr 1e-5` |
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

📌 **But their file also does something ours doesn't: it generates.** See Track B. The six defects are the price of rewriting the model; working generation is what they bought with it.

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

**A3 is the regression test.** Run it after any change to the model code, *twice*. It has earned its keep four times, and **it is step B5 of Track B.**

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

### ⚠️ The "100× smaller" headline is a PER-LAYER number — the model-level figure is much smaller for the hybrid

At 32k one attention layer's KV cache is `2 × 32768 × 4 × 128 ≈ 33.6 M` numbers; one KDA layer's recurrent state is `20 × 128 × 128 = 327,680`, **the same at 32k as at 128 tokens**. That really is ~100×, and it is the right number for one layer.

**It is the wrong number for the model**, and A6's own trap says why: *a single-layer benchmark is not a model benchmark*. The 3:1 hybrid keeps **9 of 36** attention layers, so its total inference state cannot fall below one quarter of the teacher's however good KDA is. Closed form, bf16, `num_key_value_heads = 4`, `head_dim = 128`:

| context | teacher (36 attn) | **hybrid 3:1** (9 attn + 27 KDA) | **kda_only** (36 KDA) |
|---|---|---|---|
| 2,048 | 144.0 MB | 54.1 MB — **2.7×** | 24.1 MB — **6.0×** |
| 8,192 | 576.0 MB | 162.1 MB — **3.6×** | 24.1 MB — **23.9×** |
| 32,768 | 2,304.0 MB | 594.1 MB — **3.9×** | 24.1 MB — **95.7×** |

⭐ **This is a thesis result, not a caveat.** The hybrid's nine retained attention layers are what buy its quality, and they are also what cap its memory advantage at ~4×. **The 100× belongs to the KDA-only student.** That turns "hybrid versus KDA-only" from a minor ablation into the project's central quality-versus-memory trade-off — and it is a strong argument for putting one full training run into `afm_kda_only_init`.

⚠️ **Never quote a memory ratio without saying which model it belongs to.** Per-layer, hybrid and KDA-only differ by more than an order of magnitude at 32k.

### ⚠️ B9a, corrected — the recurrent state is **fp32**, and the numbers change

**The check did its job.** After B4′ the KDA layers actually populate their state, and the state rides on the cache object as `AFMKDACache.fla` — so a naive walk of `past_key_values` returned **KV + KDA together**. The closed form, which predicts KV alone, therefore flagged `*** MISMATCH ***`. That is the instrument working, not a failure.

**And the mismatch carried a finding.** B6b measured **35.33 MB** for 27 KDA layers where the geometry predicted 18.06 MB. The ratio is not 2, so it is not a simple dtype doubling. Solving for it:

```
measured / layer = 35.33 MB / 27 = 1,372,160 B
recurrent 20 × 128 × 128 × 4 B (FLOAT32)  = 1,310,720
conv      3 × 2560 × 4 × 2 B  (bf16, width conv_size not conv_size−1) =    61,440
                                            ----------
                                             1,372,160   ✅ exact
```

⭐ **The recurrent state is kept in float32, not bf16** — sensible, since the delta-rule state accumulates over thousands of steps and bf16 would drift — **and the conv state is `conv_size` wide, not `conv_size − 1`.** Both of my assumptions were wrong. The corrected constant is **1.3086 MB per KDA layer**, and it reproduces every row of the rerun: 44.3, 53.3, 71.3, 107.3, 179.3, 323.3, 611.3 for the hybrid, and a flat 47.1 for kda_only.

✅ **Confirmed by looking, 2026-09-20.** The dtype dump reads exactly:

```
['recurrent_state'] Tensor(1, 20, 128, 128) torch.float32 = 1280.0 KB
```

`20 × 128 × 128 × 4 B = 1,310,720 B = 1280 KB`. The fp32 recurrent state is a fact, not an inference, and both closed-form checks (`check KV` and `check KDA`) now pass for all three models.

#### The corrected table — total inference state, all measured

| context | teacher | **hybrid 3:1** | **kda_only** |
|---|---|---|---|
| 512 | 36.0 MB | 44.3 MB — **0.81×** | 47.1 MB — **0.76×** |
| 1,024 | 72.0 MB | 53.3 MB — 1.35× | 47.1 MB — 1.53× |
| 2,048 | 144.0 MB | 71.3 MB — 2.02× | 47.1 MB — 3.06× |
| 8,192 | 576.0 MB | 179.3 MB — 3.21× | 47.1 MB — 12.2× |
| 32,768 | 2,304.0 MB | 611.3 MB — **3.77×** | 47.1 MB — **48.9×** |

⚠️ **The headline drops from 95.7× to 48.9×**, because the state is fp32 and the conv state is wider than I assumed. Revision 27's table was computed under a bf16 assumption and is **wrong**; this one is measured.

⭐⭐ **And the most interesting row is the first one, where the converted models use MORE memory than the teacher.** A fixed state is not a free state: it costs 1.31 MB per layer whether the context is 8 tokens or 32,768. Setting the two expressions equal:

```
teacher KV  =  36 × 2 × T × 4 × 128 × 2 B
converted   =  n_attn × 2 × T × 4 × 128 × 2 B  +  n_kda × 1,372,160 B
→ crossover at T ≈ 670 tokens, IDENTICAL for hybrid and kda_only
```

The crossover is the same for both because it is a **per-layer** property: each converted layer trades `2 × T × 512 × 2 B` of growing KV for 1,372,160 B of fixed state, and those are equal at T = 670 regardless of how many layers you convert.

**This belongs in the thesis exactly as it is.** "Below ~670 tokens the conversion costs memory; above it the saving grows without bound, reaching 3.8× for the hybrid and 49× for KDA-only at 32k." That is a sharper and more defensible claim than a single ratio, and it is the kind of detail that shows the number was measured rather than quoted.

---

### ✅ B9a — measured, 2026-09-20

| context | teacher (36 attn) | **hybrid 3:1** (9 attn) | **kda_only** (36 KDA) |
|---|---|---|---|
| 512 | 36.0 MB | 27.1 MB — 1.3× | 24.1 MB — 1.5× |
| 2,048 | 144.0 MB | 54.1 MB — 2.7× | 24.1 MB — 6.0× |
| 8,192 | 576.0 MB | 162.1 MB — 3.6× | 24.1 MB — **23.9×** |
| 32,768 | 2,304.0 MB | 594.1 MB — **3.9×** | 24.1 MB — **95.7×** |

**All three closed-form checks returned OK**, so the recursive cache walk is finding every tensor and the measured column is trustworthy.

⚠️ **Only the attention KV column is measured; the KDA state column is computed.** The 27 (or 36) KDA layers store nothing today — the adapter still hardcodes `use_cache=False` — so 18.06 MB and 24.08 MB come from the geometry (`20 × 128 × 128` plus three conv states, bf16). **They become measurable at B4′ and the measurement must reproduce exactly these figures.** Until then, say "computed" in the thesis, not "measured".

⭐ **The flat column is the result.** KDA state reads 18.06 MB at T = 512 and 18.06 MB at T = 32,768. Constant in sequence length, stated as a number rather than as an argument. That is the whole architectural claim in one row.

📌 **`peak fwd` is not the cache and must never be quoted as one.** It is 10,464 MB for the teacher and 8,160 MB for kda_only at 32k — only 22 % apart, because a full-sequence forward is dominated by activations, not by the cache. **The conversion buys almost nothing at prefill; it buys during decode, where the carried state is all that persists.** Anyone reading a memory claim will ask this, so answer it before they do.

### ⭐ B9a — the memory curve is measurable TODAY, without the bridge

A cache is built by any forward pass with `use_cache=True`; generation is only what *reads* it back. So the table above can be **measured** rather than computed, on the real checkpoints, with no Track B code: run a forward, walk the returned cache object, sum every tensor. `test_b9_cache_memory.py` does exactly that, and cross-checks the measurement against the closed form — if the two disagree, the walk missed a tensor and every number is suspect.

The 27 KDA layers currently store nothing, so their state is the analytic figure above; once B4′ lands it becomes measurable too, and the measurement should reproduce 0.67 MB per layer.

⚠️ **Do not quote `max_memory_allocated()` as "the cache".** It includes activations, which dominate a full-sequence forward at large T. The script reports it, clearly labelled, as a sanity number only.

**What still needs the bridge:** decode throughput — tokens per second with the cache in the loop. That is the half of B9 that cannot be faked.


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

**The fix — three changes, none redundant:**

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

```python
        # in AFMKDALinearAttention.forward
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is not None:
            attention_mask = None      # MANDATORY: fla 0.5.2 would overwrite ours
        elif attention_mask is not None and attention_mask.dim() != 2:
            attention_mask = None
```

⚠️ **`shape[0] == 1` is load-bearing** — fla's varlen convention flattens the batch. ⚠️ **Pinned to fla 0.5.2.** ⚠️ **And `shape[0] == 1` is exactly what B7 has to revisit**, because batched left-padded generation arrives with `shape[0] > 1` and a genuine 2-D padding mask that KDA *does* need.

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

**The fix — rename, and drop what doesn't belong:**

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

## Evaluating the student — what the protocol is, and one trap

The frozen protocol lives in `claude/eval_zeroshot_afm.sbatch` and `claude/eval_mmlu_afm.sbatch`: **ARC-Challenge, HellaSwag, WinoGrande at 0-shot; MMLU at 5-shot; no chat template**, matching RADLADS and the practical work. Metrics: `acc_norm` for ARC-C and HellaSwag, `acc` for WinoGrande and MMLU. **Do not change a shot count once a student has been measured.**

### ⚠️ Offline, a hub repo id is not a path — use the snapshot directory

`MODEL_PATH=arcee-ai/AFM-4.5B-Base` failed with `OSError: We couldn't connect to 'https://huggingface.co' … and couldn't find them in the cached files`, even though the teacher was plainly in the cache and DistillKit loads it by that exact id every run.

The reason is an environment-variable collision. `HF_HOME=$WORK_LUKAS/hf_cache` puts the hub store at `$WORK_LUKAS/hf_cache/**hub**/`, but the eval scripts also set `HUGGINGFACE_HUB_CACHE=$WORK_LUKAS/hf_cache`, which **overrides** that and points the resolver one directory too high. It looks for `hf_cache/models--arcee-ai--AFM-4.5B-Base`, finds nothing, tries the network, and is offline. DistillKit's sbatch never sets `HUGGINGFACE_HUB_CACHE`, which is why the same id works there.

Either fix works — `HUGGINGFACE_HUB_CACHE=$HF_HOME/hub`, or the explicit snapshot path:

```
MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache/hub/models--arcee-ai--AFM-4.5B-Base/snapshots/dab7922f9c868d479b365410304f466b007b1c5a
```

⭐ **Prefer the explicit snapshot path, and not only because it works.** The snapshot hash *is* a provenance record: a repo id resolves to whatever revision the cache happens to hold, and could silently resolve differently after a re-download, while `dab7922f…` pins the exact bytes that produced the number in the thesis. The practical-work scripts did this, and it was the right instinct. **Every student is a plain directory anyway**, so explicit paths are the norm everywhere else.

⚠️ **`eval-env2` does not have `flash-linear-attention`.** Irrelevant for the teacher; a hard `ImportError` inside `trust_remote_code` the first time a KDA student is evaluated. Install it there before the first student run.

---

## Next: stage 1 → `claude/first_distillkit_runs.md`

Track A is closed. **R1 was submitted 2026-09-18** at 300 M tokens, 4 nodes.

**Stage 1's job, numerically:** take the hybrid from **CE 7.94** back toward the teacher's **2.30**, targeting held-out KL ≈ **0.2**.

### Scope decision: skipping the long-context stage

> "I collapsed the first two distillation stages into a single one with frozen MLP parameters … **then allowed the long context fine tune to pick up any slack necessary in adjusting the MLP layers.**"

RADLADS's pipeline is three stages; Arcee merged 1 and 2 and used stage 3 as the place where MLPs unfreeze. **Drop it entirely and nothing outside the attention layers ever adapts** — and RADLADS reports that freezing MLPs and embeddings hurts.

| Option | What | Cost |
|---|---|---|
| **(a) Stage 1 only** | Evaluate at ≤2048, state the limitation | cheapest |
| **(b) Stage 1 + a short full-parameter stage at 2048** | Unfreezes MLPs; **RADLADS stage 2** | ~1 extra stage-1-sized run |
| **(c) All three stages** | What Arcee did | expensive |

**Recommended: (b) if budget allows.**

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

--- TRACK B: GENERATION -- full checklist lives in claude/track_b.md ---
[x] B0 / B0b. fla-Cache-everywhere FAILS on transformers 5.x; cause dropped
[x] B9a. memory curve MEASURED: 2304 / 594 (3.9x) / 24.1 MB (95.7x) at 32k
[ ] B2'-B4'. apply the patch in claude/track_b.md
[ ] B5. A3 twice -> must still be 0.000e+00
[ ] B1+B6, B7, B8, B9b -- see claude/track_b.md

--- TRACK C: GDN (designed, not built) ---
[x] gdn_installed_0.5.2.py saved as the authoritative source
[x] parameter breakdown done; the gap is expand_v + the output-gate rank
[x] framing settled: this tests KDA's fine-grained-gating claim
[x] configuration chosen: expand_v=1, use_gate=True, 20x128 (the drop-in)
[ ] AFMGDNLinearAttention adapter + `mechanism` config field
[ ] build_student.py VARIANTS entry + parameter assertion
[ ] build afm_gdn_hybrid31_init   -- AFTER R1 lands, and AFTER or BEFORE B, not during

--- OPEN, NOT BLOCKING ---
[ ] commit test_a9.py + build_student.sbatch so the tree is clean
[ ] flp layout CONFIRMED against AFM-4.5B-Base-KDA-FLP/config.json (currently OURS)
[ ] delete the leftover `_is_hf_initialized` diagnostic from test_a3.py
[x] flash-linear-attention 0.5.2 + fla-core 0.5.2 installed into eval-env2;
    torch/transformers/lm_eval unchanged (2.12.0+cu126 / 5.9.0 / 0.4.13.dev0)
[x] [AFMKDA-PROBE] prints removed from the repo (were never re-synced -- the
    module cache under $HF_HOME, not ~/.cache/huggingface, is what to clear)
[ ] note the env skew: transformers 5.8.1 (distill_env) vs 5.9.0 (eval-env2)
```

## Traps, collected

- **⚠️ Read how a configuration flag is consumed before building on what its name implies.** `functionary_packing` reads as "enable packing"; it means "assume the data is already packed in functionary's format", and does nothing without `dataset.prepacked`. Two lines in `main.py`.
- **⚠️ Never quote a memory ratio without saying which model it belongs to.** Per layer KDA is ~100x smaller at 32k; the 3:1 hybrid is ~3.9x because it keeps 9 attention layers; kda_only is ~96x. Same architecture, three different honest numbers.
- **⚠️ Separate what you observed from what you inferred.** B0 observed three identical failures in transformers' wrapper; the *reason* is a hypothesis, and acting on the hypothesis before reading the source would have thrown away the simpler design for nothing.
- **⚠️ A ten-minute probe can retire a day of work.** B0 cost one debug-queue job and killed an approach that would have taken B2, B3 and B4 to discover was impossible. Probe the assumption the plan rests on, first.
- **⚠️ The remote-code module cache is under `$HF_HOME`, not `~/.cache/huggingface`.** Clearing the wrong one is silent: the model loads and runs the old code.
- **⚠️ `save_pretrained` DOES copy remote-code `.py` files into a checkpoint.** Look in the directory before writing a workaround for a problem that isn't there.
- **⚠️ The reference targets transformers 4.57 and we run 5.x.** That gap has now changed the answer six times — `_init_weights`, the causal mask, checkpointing kwargs, `auto_map`, `@strict` configs, and now the whole cache design.
- **⚠️ Check what a guard sees on every path that reaches it, including evaluation.** A `use_cache` check written for training would have fired on R1's first eval pass and killed a four-hour run. **A safety check that fires on a legitimate path is worse than no check.**
- **⚠️ The slow, obviously-correct workaround is usually the oracle for testing the fast fix.** `use_cache=False` generation is too slow to ship and exactly right as B1's ground truth.
- **⚠️ A capability you cannot exercise is a claim you cannot make.** KDA's 100×-smaller inference state is the point of the whole project and is unmeasurable until generation works.
- **A test that supplies the input it assumes confirms the assumption, not the system.** A7 → A7b → A7c → R0 → R0b → A9: each hop moved the question closer to production and each hop found something.
- **⚠️ A probe that can only report success tells you nothing when it fails.** Instrument the decision, not the happy path.
- **⚠️ stdout and stderr are two different files under SLURM.** B0 printed `RAISED:` to stdout and the traceback to stderr, so the `.out` file recorded the failure and hid the reason. `traceback.print_exc(file=sys.stdout)`.
- **⚠️ A discriminating test needs an input the two branches can disagree on.** B0's cached-vs-uncached control used sixteen arbitrary token ids; the difference was real but only 4%, because on gibberish the next token is dominated by the last one. Use real text, a few hundred tokens.
- **`X.__dict__` answers "defined here", not "defined by this library".** B0's `own=` column read a thin alias subclass and reported that fla defines nothing.
- **⚠️ A cache can lie about its length.** `get_seq_length()` reads layer 0, and in a hybrid layer 0 may store nothing — so `generate()` is told the cache is empty no matter what the attention layers hold.
- **Count the calls before trusting a patch.** *A patch that is installed is not a patch that runs.*
- **⚠️ A `def` one indent too deep is a silent no-op.** Assert `"forward" in Cls.__dict__` at import.
- **Two identical numbers are a finding**, stronger than a merely-large number. It has meant "the mask is ignored entirely" three times — A7b, the vanished `forward`, and A9.
- **Always include a negative control.**
- **The same value can travel under two names.** The A7c probe showed `cu_seqlens` already arriving, unrecognized — **look at what is already there before building a second path.**
- **⚠️ `HUGGINGFACE_HUB_CACHE` overrides `HF_HOME`'s `hub/` subdirectory.** Set one too high and every offline repo-id load fails with a *network* error that says nothing about paths. **Prefer an explicit snapshot path — it is also a provenance record.**
- **Do not assume one library layer's quirks carry to its sibling.** KDA 0.5.2 overwrites a supplied `cu_seqlens`; GDN 0.5.2, same release, does not.
- **Make the build record a number you can check by hand.** `loading_info` 441/108 re-verifies the whole inventory for free on every rebuild.
- **`dict.get(key, default)` turns a wrong key into a plausible answer.** Print the whole object first.
- **A library file you downloaded is not the library you installed.**
- **fla 0.5.2 overwrites `cu_seqlens` whenever a mask is present.** Supply one *or* the other.
- **For a remote-code model, `config.json` is the contract; the class is decoration.**
- **HF copies only ONE level of relative imports.** Clear the module cache after any `.py` change *or rename*.
- **The `.py` files in a checkpoint directory are copies** — which is why editing the repo is safe while a job is queued, and why the re-sync loop is not.
- **A variable assigned inside an sbatch script beats `--export`.** Use `${VAR:-default}`.
- **A dirty tree makes a recorded hash a half-truth.**
- **Never override `_init_weights` on a subclass of a library model in transformers 5.**
- **Run any suspect test twice.**
- **If a test needs a tolerance you invented, redesign the test.**
- **A NaN can be the answer, not a broken test.**
- **GQA expansion is group-major.** Query head `h` uses KV head `h // n_rep`.
- **transformers 5.x configs are `@strict` dataclasses.**
- **`_can_record_outputs` must name your subclassed layer**, or `hs_cosine` silently compares nothing.
- **`re.search` matches anywhere; `re.match` only at position 0.**
- **`use_qk_l2norm_in_kernel=True` discards transferred q/k magnitude.**
- **The reference's layer indices are 1-based; ours are 0-based.**
- **Never splat a config dict into a fla layer constructor.** The trailing `**kwargs` swallows misspellings.
- **RoPE has no learned parameters.** **36 layers, 37 hidden states.**
- **A single-layer benchmark is not a model benchmark.** A6's 0.31× became **0.88×** at the model level.
- **KDA's memory advantage is an inference property**, not a training one — which is exactly why Track B exists.
- **A checkpoint's directory name is a label; `config.json` is the definition.**
- **Two independent measurements agreeing is worth more than either alone.** A5b's 7.9412 and R0's 7.918.

## Sources

- `distillkit_blogpost_kda.txt` — [Distilling Kimi Delta Attention into AFM-4.5B](https://www.arcee.ai/blog/distilling-kimi-delta-attention-into-afm-4-5b-and-the-tool-we-used-to-do-it), Charles Goddard, 2025-12-15
- `RADLADS.pdf` — three-stage pipeline; de-novo-initialization and frozen-MLP ablations; **the 0-shot / 5-shot-MMLU evaluation convention**. ⚠️ **Says nothing about packing or document boundaries**
- `kda_installed_0.5.2.py` and `gdn_installed_0.5.2.py` — **the authoritative fla sources**
- `modeling_arcee_kda.py` — the reference implementation; ⭐ **and the working template for Track B's fla `Cache` bridge**
- `main.py`, `trainer.py`, `signals.py`, `monkey_patch_packing.py`, `pack_logits.py` (DistillKit)
- `claude/eval_zeroshot_afm.sbatch`, `claude/eval_mmlu_afm.sbatch` — the frozen evaluation protocol
- [arcee-ai/AFM-4.5B-Base config.json](https://huggingface.co/arcee-ai/AFM-4.5B-Base/blob/main/config.json) · [AFM-4.5B-Base-KDA-NoPE](https://huggingface.co/arcee-ai/AFM-4.5B-Base-KDA-NoPE)