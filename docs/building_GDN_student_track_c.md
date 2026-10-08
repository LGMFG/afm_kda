# Track C — Building the GDN Student

*Companion to `claude/building_the_student.md` (Track A — how the KDA student was built), `claude/track_b.md` (generation, closed) and `claude/first_distillkit_runs.md` (the live run log). This document covers the **second mechanism**: Gated DeltaNet.*

*Revision 9 (2026-09-25) — ✅ **Track C's test ladder is complete** (§8h). **B7c** on the re-synced trained checkpoint: fluent on all three rows, `use_cache` True, most-padded row best — C-5 finally exercised through `generate()` on real KDA layers. **B6e at 128 steps**: the fp32 curve is **flat**, not merely bounded. **C-7** landed, 6/6 markers on both trained directories. Committed at **`8f40624`**, clean tree. What remains is the rebuild, A3/C1, and the FLP layout question.*

*Revision 8 (2026-09-25) — ✅✅ **C-5 passes; §8f is resolved in §8g.** Control A (batch 1, **zero padding**) reads **6.8 ulp**, so most of the magnitude exists before any pad token does. Control B (the **untrained KDA hybrid**) reads 8.1/12.9/12.0/12.0 with its **most-padded row its cleanest** — the *opposite* ordering to the GDN student's, on the same stash loop, which no contamination story survives. **The GDN student is also better than the KDA one on every statistic.** ⭐ The real finding is a baseline: **untrained students sit at 6–13 ulp and 27–30/32** where a trained one sits at 1–2.5 and 31–32/32, and B7's published pass band does not transfer between them.*

*Revision 7 (2026-09-25) — 🔶 **both B7 runs landed and neither is interpretable yet** (§8f). `afm_kda_only_init` shows **no padding ordering whatsoever** — the *unpadded* row is second-worst — which is the all-linear mask path behaving. `afm_gdn_hybrid31_init` leans the wrong way by ~1.7×, but ⚠️⚠️ **B7's default `LENS` makes padding and length perfectly anti-correlated**, so that ordering cannot be read, and ⚠️ both students are **untrained**, where the only B7 baseline on record is a trained one. **Two one-job controls settle both**: `LENS=120,120,120,120` on the GDN hybrid, and plain B7 on `afm_kda_hybrid31_init`.*

*Revision 6 (2026-09-25) — ✅ **`afm_gdn_only_init` built** (§8e): 5,237,399,968, 468/144, 36 GDN + 0 attention, 9.8 GiB, every closed form exact. ⚠️⚠️ **And §C-5's named test was wrong.** It is **B7, not B7c**, and the subject is **`afm_gdn_hybrid31_init`, under `IMPL=sdpa`** — B7c's criterion is fluent prose, so it needs a trained model, and the only trained checkpoint carries `3aafffc`'s frozen `.py` files, which predate every Track C edit. B7 is teacher-forced and therefore works on an untrained student. ⭐ The GDN arm of the stash loop is what has never run; the KDA arm is provably unchanged.*

*Revision 5 (2026-09-25) — ✅ **B6e run on `afm_kda_only_init`, the all-linear student: clean.** fp32 worst 1.439e-02 against the hybrid's 8.021e-03, argmax 32/32, and **growth with position unchanged** (8.1× vs 9.1×), so a 36-layer stack with no attention anchors drifts no worse than the hybrid. ⚠️ Two corrections in §8c: the script's printed 1e-5 verdict threshold **predates the TF32 finding and would fail the validated hybrid**, and — my error when proposing the run — **B6 cannot validate the `attn_layer_idx is None` fallbacks**, because in an all-linear model nothing consumes their answer. They are *inert*, not *validated*. ⬜ **B7 on an `only` student is the job that actually gates one**, and the 128-step boundedness check now has its subject.*

*Revision 4 (2026-09-25) — ✅ **C1 passes, and the drop-in worked.** At initialisation the GDN student reads **CE 7.9247** against the KDA student's **7.9269** — a gap of **0.0022 nats, an order of magnitude below the ~0.02 seed noise** already recorded for "the untrained hybrid". **The two mechanisms are indistinguishable at the starting line**, which is exactly what matching projection geometry and initialisation was supposed to buy: any difference after training is attributable to the mechanism, not to where it started. Forward and backward clean, `543/543` tensors with gradients — itself a closed form. ⬜ **Newly visible gap:** the all-linear cache path (`attn_layer_idx is None`) has never been tested for *either* mechanism, and it gates every `only` variant.*

*Revision 3 (2026-09-23) — ✅ **`afm_gdn_hybrid31_init` is built.** 5,141,828,536 parameters, `missing/unexpected` **360 / 108**, 27 GDN + 9 attention, 9.6 GiB, `"dirty": false`. A3 still reads `0.000e+00`, so the KDA path is untouched. **Every closed form in this document now has a measurement behind it.** ⭐ Also corrects §3's claim that the 5.29 M per-layer difference "is exactly the output gate" — it is the net of **+5,895,680** for GDN's full-rank output gate and **−604,160** for its per-head forget gate, and that second number is the price of the fine-grained gating KDA's paper claims as its advance. Decomposition in §4.*

*Revision 2 (2026-09-23) — ⭐⭐ **the geometry is now MEASURED, not computed.** A four-layer throwaway model at full width confirms **32,901,288 parameters in 13 tensors** per GDN layer and **27,612,308 in 16** per KDA layer, so every total in §4 follows arithmetically from a measurement. `dt_bias` came back `(20,)` for GDN against `(2560,)` for KDA, confirming the per-head / per-channel difference in the model rather than in the source. ⚠️ **Two traps found while doing it**: `@strict` configs do **not** validate `from_pretrained` kwargs — transformers applies them *after* `__post_init__`, so the mechanism guard had to move to model construction; and the decoder-layer dispatch must **raise** on an unknown mechanism rather than falling back to KDA, or a typo silently builds a KDA student carrying a GDN label.*

*Revision 1 (2026-09-23) — written after reading `configuration_afm_kda.py`, `modeling_afm_kda.py`, `kda_installed_0.5.2.py` and `gdn_installed_0.5.2.py` line by line. ⭐ **Two things changed from the earlier sketch**: the config needs **one** new field, not six — `kda_layers` already says "which layers are linear" and is orthogonal to *which* mechanism fills them; and the padding-mask stash loop in `AFMKDAModel.forward` matches on a class **name string**, which is the one place a GDN student would silently lose Track B's B7 fix.*

---

## 0. ⭐ No new files — and why that is the right call

There is **no `configuration_afm_gdn.py` and no `modeling_afm_gdn.py`.** Everything goes into the files that already exist, and the GDN student is selected by one config field.

```
afm_kda/
  configuration_afm_kda.py    + mechanism, gdn_expand_v, gdn_use_gate
  modeling_afm_kda.py         + AFMGDNLinearAttention   (~45 lines)
                              + one branch in AFMKDADecoderLayer.__init__
                              + one name in the stash loop
                              + one elif in init_new_parameters
  build_student.py            + GDN_DELTA, 3 VARIANTS rows, mechanism threading
  build_student.sbatch        VARIANT=${VARIANT:-hybrid31}
```

**Three reasons, in order of how expensive it would be to get this wrong.**

**1. A separate file would have to duplicate almost everything.** The GDN student needs `AFMKDACache`, `AFMKDAModel`, `AFMKDAAttention`, `AFMKDADecoderLayer`, `init_new_parameters` and `transfer_attention_into_kda` — every one of which is mechanism-agnostic already. A parallel module either copies ~400 lines (two versions of the Track B cache bridge, drifting apart the first time one is fixed) or imports them from `modeling_afm_kda.py`, in which case the second file holds one class and exists for no reason.

**2. `auto_map` is baked into every saved checkpoint.** `config.json` says `"AutoModelForCausalLM": "modeling_afm_kda.AFMKDAForCausalLM"`, and HF copies that file into the checkpoint directory. Renaming or splitting means a second `auto_map`, a second `CODE_FILES` list, and a second import closure to keep correct — for a model that differs from the first by forty lines.

**3. It is the experiment.** The comparison is supposed to be KDA against GDN with *everything else held fixed*. Two file trees make "everything else" a claim you have to maintain; one file tree makes it structural. The only things that can differ are the adapter and the two config fields.

⚠️ **So `modeling_afm_kda.py` becomes a misnomer**, exactly like `kda_layers`. Same reason, same decision: renaming breaks every checkpoint that points at it. Note it in the thesis once — *the module is named for the first mechanism it implemented* — and move on.

📌 Think of it as one model family with a pluggable token-mixer: `AFMKDAAttention` (softmax), `AFMKDALinearAttention` (KDA), `AFMGDNLinearAttention` (GDN), chosen per layer by `kda_layers` and per model by `mechanism`.

---

## 1. Why GDN, and what question it answers

Both KDA and Gated DeltaNet are gated delta-rule linear-attention layers, so at first glance the comparison looks narrow. **It isn't** — look at how each computes its forget gate, in the installed 0.5.2 source:

| | forget gate | shape | granularity |
|---|---|---|---|
| **KDA** | `f_proj`: `Linear(2560,128)` → `Linear(128,2560)` | `[B,T,H,K]` | **per channel** — 2,560 decay values per token |
| **GDN** | `a_proj`: `Linear(2560, num_v_heads)` | `[B,T,H]` | **per head** — 20 scalars per token |

Visible in the kernel calls themselves: KDA passes `g=` a tensor rearranged to `'... (h d) -> ... h d'`; GDN passes `g=self.a_proj(hidden_states)` unrearranged.

⭐ **Fine-grained channel-wise gating is precisely what the Kimi Delta Attention paper claims as its advance over Gated DeltaNet.** So KDA-vs-GDN here is a direct test of that claim **in a setting its authors did not test** — distillation from a pretrained transformer rather than training from scratch. That is a research question, not a horse race, and it is what makes GDN the right second mechanism rather than an arbitrary one.

📌 **And Track B closing makes the comparison cheap.** The cache bridge, the `get_mask_sizes` override and the padding-mask stash are all mechanism-agnostic: they live in `AFMKDACache` and `AFMKDAModel`, not in the KDA adapter. Track C inherits every one of them.

---

## 2. The two fla layers, compared — verified 2026-09-23

Read side by side from `kda_installed_0.5.2.py` and `gdn_installed_0.5.2.py`.

### What is the same

Both assert a 2-D padding mask, word for word. Both switch kernel by length — `mode = 'fused_recurrent' if (q_len <= 64 and not self.training) else self.mode` — so **the chunk/fused_recurrent bit-identity caveat from B6d applies to GDN unchanged**. Both call `get_layer_cache` / `update_layer_cache` with `offset=q_len`. Both use `use_qk_l2norm_in_kernel=True`, so **the A5b caveat that transferred q/k magnitude is discarded applies to GDN too** — only direction survives. Both keep `recurrent_state` in fp32.

### What is different — and it matters

| | KDA 0.5.2 | GDN 0.5.2 |
|---|---|---|
| unpad guard | `if attention_mask is not None:` | `if cu_seqlens is None and attention_mask is not None:` |
| repad guard | `if attention_mask is not None:` | `if attention_mask is not None:` ← **not** guarded on `cu_seqlens` |
| supplying **both** `cu_seqlens` and a mask | mask wins; **our `cu_seqlens` is silently overwritten** | unpad skipped so `indices` is never bound, repad runs → **`NameError: indices`** |
| forget gate | per channel, via a 128-dim bottleneck | per head, a single `Linear(hidden, num_v_heads)` |
| output gate | `g_proj`: 2560→**128**→2560 (+bias), low rank | `g_proj`: 2560→value_dim, **full rank** |
| `A_log` init | `log(uniform(1, 16))` | **`log(uniform(0, 16))`** |
| `dt_bias` shape | `[gate_dim]` = **[2560]**, per channel | `[num_v_heads]` = **[20]**, per head |
| fused q/k/v conv fast path | absent | `_use_fused_qkv_conv`, fires when `last_state is None and not use_cache and cu_seqlens is None` |
| `use_gate=False` option | absent | present; swaps `FusedRMSNormGated` for plain `RMSNorm` |

⭐ **One line covers the first row of danger**, and the KDA adapter already has it:

```python
cu_seqlens = kwargs.get("cu_seqlens")
if cu_seqlens is not None:
    attention_mask = None
```

For KDA that prevents a silent wrong answer. For GDN it prevents a crash. **Same guard, two different disasters** — which is why it must be copied into the GDN adapter verbatim and not "simplified".

📌 **`_use_fused_qkv_conv` fires during our training.** The collator sends no mask, gradient checkpointing forces `use_cache=False`, and no `cu_seqlens` is delivered (Part 4 of the runs doc) — so all three conditions hold. It is a performance path, not a semantic one, but **GDN training exercises code KDA never does**, which is worth knowing if a throughput comparison ever looks surprising.

⚠️ **`A_log` and `dt_bias` differ in both distribution and shape.** Reusing KDA's initialisation for GDN would be wrong twice over — `uniform(1,16)` instead of `uniform(0,16)`, and a 2560-wide draw copied into a 20-wide parameter. `init_new_parameters` needs its own branch.

---

## 3. The configuration — one new field, and the variants come free

**Yes: the existing flags carry over unchanged**, and this is worth stating precisely because it is what makes the experiment clean.

```python
kda_layers: list[int] | None = None   # 0-BASED; None/[] => pure attention
kda_num_heads: int = 20               # 20 x 128 = 2560 = hidden_size
kda_head_dim: int = 128
kda_conv_size: int = 4
use_rope: bool = True
use_attn_gate: bool = False
```

⭐ **`kda_layers` does not say "KDA". It says "which layers are linear."** The layout and the mechanism are orthogonal, so the three variants are a cross product and nothing about them needs duplicating:

| | `kda` | `gdn` |
|---|---|---|
| **hybrid31** — 27 linear, attention at 3,7,…,35 | `afm_kda_hybrid31_init` ✅ built | `afm_gdn_hybrid31_init` ⬜ **the run that matters** |
| **only** — all 36 linear | `afm_kda_only_init` ✅ built | `afm_gdn_only_init` ⬜ |
| **flp** — attention at 0,1,2,3 + 5 more | `afm_kda_flp_init` ✅ built | `afm_gdn_flp_init` ⬜ |

`kda_num_heads`, `kda_head_dim` and `kda_conv_size` map one-to-one onto GDN's `num_heads`, `head_dim` and `conv_size` with identical meaning. `use_rope` and `use_attn_gate` describe the **attention** layers and are mechanism-independent by construction.

⚠️ **The `kda_` prefix is now a misnomer, and it is kept deliberately.** Renaming to `linear_*` would change every saved `config.json` — including `output/r1_stage1_hybrid31`, which is the trained checkpoint every number in the thesis so far comes from. **A field name is not worth breaking a checkpoint over.** Say so once in the thesis and move on.

⭐ **There is a second, better reason to share the geometry fields: it makes the drop-in property structural.** With one set of head/dim/conv values, the KDA and GDN students *cannot* accidentally be given different geometry. The only differences that survive are the two the study is about.

### What genuinely has to be added

Only what has no KDA counterpart:

```python
    # --- Track C: which linear-attention mechanism fills the non-attention layers
    mechanism: str = "kda"          # "kda" | "gdn"
    gdn_expand_v: float = 1.0       # fla defaults to 2.0; 1.0 keeps value_dim at 2560
    gdn_use_gate: bool = True       # GDN-only; KDA's output gate is not optional
```

`mechanism` defaults to `"kda"`, so **every existing config.json still builds exactly the model it built before** — the four rebuilt students and R1's checkpoint included.

### The configuration decision, and why

| Config | per layer | student | A5b transfer | character |
|---|---|---|---|---|
| **`expand_v=1`, `use_gate=True`, 20×128** ⬅ **chosen** | 32.90 M | 5.142 B | ✅ **works unchanged** | drop-in; every projection 2560×2560 |
| `expand_v=2`, GDN's own 0.75·h recommendation | ~39.4 M | ~5.32 B | ❌ shapes don't fit | GDN as its authors intend it |
| `expand_v=1`, `use_gate=False` | ~26.3 M | ~4.98 B | ✅ works | closest parameter match, but removes GDN's gate |

**Take the drop-in.** Not because it is parameter-matched — it isn't, 32.90 M against KDA's 27.61 M — but because it keeps *everything else identical*: same projection shapes, same weight transfer, same initialisation scheme, same adapter pattern. **An initialisation difference would be a far bigger confound than 19 % more mixing parameters.**

⭐ **And the residual gap is itself the finding** — see §4 for the exact decomposition, which is the *net of two opposite-signed terms* and not, as an earlier revision of this section said, simply the output gate. Report both parameter counts and say you matched **projection geometry and initialisation**, not parameter count.

📌 GDN's own recommendation (`0.75·hidden`, `expand_v=2`) exists so a GDN layer costs `6·hidden²`, **matching Mamba2** — it is itself a budget-matching convention anchored to a different reference. Deviating from it to match KDA instead is the same kind of move, not a violation.

---

## 4. The geometry, worked out

`num_heads=20, head_dim=128, expand_v=1, use_gate=True` → `head_k_dim = head_v_dim = 128`, `key_dim = value_dim = 2560`.

| tensor | n | params |
|---|---|---|
| `q_proj`, `k_proj`, `v_proj`, `o_proj`, **`g_proj`** — all 2560×2560 | 5 | 32,768,000 |
| `a_proj`, `b_proj` — 2560×20 each | 2 | 102,400 |
| `q_conv1d`, `k_conv1d`, `v_conv1d` — [2560, 1, 4] | 3 | 30,720 |
| `A_log`, `dt_bias` — [20] each | 2 | 40 |
| `o_norm` — [128] | 1 | 128 |
| **per GDN layer** | **13** | **32,901,288** |
| Arcee attention layer | | 15,728,640 |
| **delta per converted layer** | | **+17,172,648** |
| attention gate (`gate_proj`) | | +6,553,600 |

✅ **MEASURED 2026-09-23**, not computed — a four-layer model at full width returns **32,901,288 in 13 tensors** for GDN and **27,612,308 in 16** for KDA, both exact. `dt_bias` is `(20,)` for GDN against `(2560,)` for KDA, which is the per-head / per-channel gate difference showing up in the model rather than only in the source. **So the totals below are arithmetic on a measurement.**

**The three variants, closed form:**

| variant | linear | attn | parameters | missing | unexpected |
|---|---|---|---|---|---|
| `gdn_hybrid31` | 27 | 9 | **5,141,828,536** | 27×13 + 9 = **360** | 27×4 = **108** |
| `gdn_flp` | 27 | 9 | **5,141,828,536** | **360** | **108** |
| `gdn_only` | 36 | 0 | **5,237,399,968** | 36×13 = **468** | 36×4 = **144** |

*(For comparison: `kda_hybrid31` is 4,999,026,076 with 441/108, and `kda_only` is 5,046,996,688.)*

✅ **CONFIRMED BY THE BUILD, 2026-09-23** (job 58447265): `parameters: 5,141,828,536`, `missing keys: 360   unexpected keys: 108`, `layer types: 27 GDN, 9 attention`, 9.6 GiB on disk, `"dirty": false`. **Every closed form in this document now has a measurement behind it**, including the two the four-layer probe could not reach — the missing/unexpected counts depend on how `from_pretrained` matches names, and 360 = 27×13 + 9 while 108 = 27×4 land exactly.

### ⭐ Where the 5,288,980 per layer actually goes

An earlier revision said this difference "is exactly GDN's full-width output gate". That was loose — it is the **net of two opposite-signed terms**, and the decomposition is the more useful statement because *both* terms are things the thesis is studying:

| component | KDA | GDN | Δ |
|---|---|---|---|
| q/k/v/o_proj | 26,214,400 | 26,214,400 | 0 |
| **output gate** | 657,920 — 2560→**128**→2560 | 6,553,600 — 2560→2560 | **+5,895,680** |
| **forget gate** | 655,360 — **per channel** | 51,200 — **per head** | **−604,160** |
| `dt_bias` | 2,560 | 20 | −2,540 |
| `b_proj`, convs, `A_log`, `o_norm` | 51,200 + 30,720 + 20 + 128 | identical | 0 |
| **total** | **27,612,308** | **32,901,288** | **+5,288,980** |

⭐ **GDN pays ~5.90 M for a full-rank output gate and gets ~0.60 M back by making the forget gate per-head instead of per-channel.** That second row — **604,160 parameters per layer, 16.3 M across the student** — *is* the price of the fine-grained gating that the Kimi Delta Attention paper claims as its advance. The comparison therefore asks: is 604 k parameters per layer of channel-wise decay worth more than 5.90 M of output-gate rank?

✅ Cross-checked against the build: `5,141,828,536 − 4,999,026,076 = 142,802,460`, and `142,802,460 ÷ 27 = 5,288,980` exactly.

---

## 5. The patch — six edits, in dependency order

### C-1. `configuration_afm_kda.py`

Below `use_attn_gate`:

```python
    # --- Track C: which linear-attention mechanism fills the non-attention layers
    mechanism: str = "kda"          # "kda" | "gdn"
    gdn_expand_v: float = 1.0       # fla defaults to 2.0; 1.0 keeps value_dim at 2560
    gdn_use_gate: bool = True       # GDN-only; KDA's output gate is not optional
```

and in `__post_init__`, after `self.auto_map = dict(type(self).auto_map)`:

```python
        if self.mechanism not in ("kda", "gdn"):
            raise ValueError(
                f"mechanism must be 'kda' or 'gdn', got {self.mechanism!r}")
```

⚠️⚠️ **That guard catches the file, not the call — measured 2026-09-23.** `AFMKDAConfig(mechanism="gnd")` raises, but `AFMKDAConfig.from_pretrained(TEACHER, mechanism="gnd")` **does not**: transformers builds the config from the JSON, runs `__post_init__` (which validates the *default*, `"kda"`, and passes), and only then writes the caller's kwargs on by attribute assignment. All validation has already run by then.

**This is a general fact about this config class, not a quirk of `mechanism`.** Nothing validates `kda_layers`, `use_rope` or any other field passed as a kwarg either; they simply have never been given a bad value. **Any invariant that must hold has to be checked where the value is used**, which is why the mechanism guard is repeated in `AFMKDAModel.__init__` (C-4b).

Optionally, to stop the naming debt spreading:

```python
    def is_linear_layer(self, layer_idx: int) -> bool:
        """0-based.  True for a KDA *or* GDN layer -- whichever mechanism is set.

        `kda_layers` names WHICH layers are linear, not WHAT fills them.  The
        field keeps its name because renaming it would invalidate every saved
        config.json, including the trained checkpoint.
        """
        return layer_idx in self.kda_layers

    is_kda_layer = is_linear_layer          # the name the rest of the code uses
```

### C-2. `modeling_afm_kda.py` — the import

Below the `KimiDeltaAttention` import:

```python
try:
    from fla.layers.gated_deltanet import GatedDeltaNet
except ImportError as e:
    raise ImportError("pip install -U 'flash-linear-attention[cuda]'") from e
```

### C-3. The adapter — directly after `AFMKDALinearAttention`

```python
class AFMGDNLinearAttention(nn.Module):
    """Adapts fla's GatedDeltaNet to ArceeDecoderLayer's call contract.

    ⚠️ The mask/cache preamble below is DUPLICATED from AFMKDALinearAttention
    on purpose.  Factoring it into a shared base would change the KDA path,
    which is validated (A3, B6, B7, B7c) and is the code the trained
    checkpoints run.  If you change one, change the other, and re-run A3.
    """

    def __init__(self, config: AFMKDAConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # Every argument by name.  GDN's trailing **kwargs swallows misspellings
        # exactly as KDA's does.  Geometry is SHARED with KDA on purpose: the
        # drop-in property is what makes the comparison about the mechanism.
        self.attn = GatedDeltaNet(
            hidden_size=config.hidden_size,
            num_heads=config.kda_num_heads,       # shared geometry -- see C-1
            head_dim=config.kda_head_dim,
            conv_size=config.kda_conv_size,
            expand_v=config.gdn_expand_v,         # GDN-only
            use_gate=config.gdn_use_gate,         # GDN-only
            mode="chunk",                         # fla asserts chunk during training
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states,
        position_embeddings=None,   # accepted and ignored: GDN carries position implicitly
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is not None:
            # ⚠️ MANDATORY for GDN, not merely advisable as it is for KDA.
            # GDN guards its UNPAD with `cu_seqlens is None` but NOT its REPAD,
            # so supplying both leaves `indices` unbound -> NameError.
            attention_mask = None
        elif attention_mask is None or attention_mask.dim() != 2:
            # super().forward() already turned the 2-D padding mask into a 4-D
            # causal bias; fla asserts on anything but 2-D and needs the real
            # padding mask for its unpad/repad path.   (Track B, defect 2)
            attention_mask = getattr(self, "_padding_mask_2d", None)

        # ---- B4': take the recurrent cache off the carrier ---------------
        fla_cache = getattr(past_key_values, "fla", None)
        want_cache = bool(kwargs.get("use_cache")) and fla_cache is not None

        if (fla_cache is None and past_key_values is not None
                and past_key_values.get_seq_length() > 0):
            raise NotImplementedError(
                "AFMGDNLinearAttention was asked to continue from a populated "
                "cache that carries no fla state. This would silently produce "
                "wrong output. Use AFMKDACache, or use_cache=False."
            )

        out, _, _ = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=fla_cache,
            use_cache=want_cache,
            cu_seqlens=cu_seqlens,
        )
        return out, None            # no attention weights exist for a recurrent layer
```

### C-4. `AFMKDADecoderLayer.__init__` — choose the mechanism, and **raise** on anything else

```python
        if config.is_kda_layer(layer_idx):
            if config.mechanism == "kda":
                self.self_attn = AFMKDALinearAttention(config, layer_idx)
            elif config.mechanism == "gdn":
                self.self_attn = AFMGDNLinearAttention(config, layer_idx)
            else:
                raise ValueError(f"unknown mechanism {config.mechanism!r}")
        else:
            self.self_attn = AFMKDAAttention(config, layer_idx)
```

⚠️ **Not `else: AFMKDALinearAttention`.** With a fallback, a typo'd mechanism builds a **KDA student carrying a config that says otherwise** — it runs, it trains, it benchmarks, and it is mislabelled for the rest of the project. An unknown mechanism must never silently produce a valid model.

### C-4b. `AFMKDAModel.__init__` — the guard that actually fires

Because of the `@strict` ordering above, this is the first point where a kwarg-supplied mechanism can be checked. ⚠️ **Keep the rest of the method** — the two lines that replace `self.layers` are what make this a hybrid at all:

```python
    def __init__(self, config: AFMKDAConfig):
        # ⚠️ The __post_init__ check does NOT catch a mechanism passed as a
        # from_pretrained kwarg -- transformers applies those AFTER __post_init__.
        # This is the first point where a wrong value can be caught.
        if config.mechanism not in ("kda", "gdn"):
            raise ValueError(
                f"mechanism must be 'kda' or 'gdn', got {config.mechanism!r}")
        super().__init__(config)
        self.layers = nn.ModuleList(
            [AFMKDADecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.post_init()
```

✅ Verified firing: a config built valid and then reassigned — exactly what `from_pretrained` does internally — is rejected at model construction.

### C-5. ⚠️ The padding-mask stash — the one edit that touches shared code

`AFMKDAModel.forward` currently ends with:

```python
        for layer in self.layers:
            if type(layer.self_attn).__name__ == "AFMKDALinearAttention":
                layer.self_attn._padding_mask_2d = raw2d
```

**It matches on a class-name string.** A GDN student would sail past it, `_padding_mask_2d` would never be set, `getattr(..., None)` in the adapter would return `None`, and **every padded row would silently feed pad tokens into the recurrent state** — the exact bug Track B spent a week finding, reintroduced by a string comparison.

```python
        for layer in self.layers:
            if type(layer.self_attn).__name__ in (
                    "AFMKDALinearAttention", "AFMGDNLinearAttention"):
                layer.self_attn._padding_mask_2d = raw2d
```

⚠️ **Re-run A3 afterwards — but know what it does and does not cover.** The change is provably behaviour-preserving for KDA (a tuple-membership test whose first element is the old string), and **A3 cannot exercise this line at all**: A3's student has all three flags at AFM's defaults, so `kda_layers` is empty and there are no linear layers for the loop to touch. What A3 actually gates here is that the module still **imports** with the new `GatedDeltaNet` dependency, that the config changes did not break construction, and that the new guard in `AFMKDAModel.__init__` does not reject a valid config. ⚠️⚠️ **And the test named here was wrong — corrected 2026-09-25.** An earlier revision said B7c. It is **B7**, and the subject is the **GDN** student, for two reasons:

- **B7c's pass criterion is fluent English**, so it needs a *trained* model. The only trained model is `output/r1_stage1_hybrid31`, and a checkpoint carries its own frozen `.py` copies — R1's are from **`3aafffc`**, which predates every Track C edit. Pointing B7c at it measures the **old** stash loop. **B7 is teacher-forced**: it compares the model against *itself*, padded-batch against per-row unpadded, so an untrained student's gibberish is irrelevant and the comparison is still exact in principle.
- **The KDA half of C-5 is provably unchanged** (a tuple whose first element is the old string). What is genuinely new, and has never run, is the **`AFMGDNLinearAttention`** arm — and `afm_gdn_hybrid31_init` is the only checkpoint that carries both the current code and 27 linear layers.

```bash
sbatch --export=ALL,MODEL=$WORK_LUKAS/students/afm_gdn_hybrid31_init,IMPL=sdpa \
       $WORK_LUKAS/afm_kda/test_b7_batched.sbatch
```

⚠️ **`IMPL=sdpa` is not decoration.** Under `flash_attention_2` `create_causal_mask` short-circuits, the mask reaches the linear layers 2-D by itself, and **the stash is never exercised** — the trap already recorded in `track_b.md` §4. A green fa2 run would say nothing about C-5.

⭐ **And a C-5 failure on GDN is not subtle in the source.** With `_padding_mask_2d` never set, the adapter passes `attention_mask=None`; `gdn_installed_0.5.2.py` then skips `get_unpad_data`, never builds `indices`, skips `pad_input` — and runs the whole padded row, pad tokens included, straight into the recurrent state. No assert, no exception, wrong logits on every padded row. That is Track B's Defect 2 reproduced exactly, which is why B7's **ordering** check is the right read: the most-padded row should be among the *cleanest*, not the worst.

📌 *Everything else in `AFMKDAModel` and `AFMKDACache` is already mechanism-agnostic and needs no change: `_first_attn_layer_idx` tests `isinstance(layer.self_attn, AFMKDAAttention)`, the cache install tests nothing about the linear layers, and `get_seq_length` / `get_mask_sizes` answer from the attention index.*

### C-6. `init_new_parameters` — GDN's own scheme

The loop currently reads `if not isinstance(module, KimiDeltaAttention): continue`. Restructure so both layer types are reached:

```python
    for name, module in model.named_modules():
        if isinstance(module, KimiDeltaAttention):
            if f"{name}.A_log" in missing:
                module.A_log.copy_(
                    torch.empty_like(module.A_log).uniform_(1, 16).log())
            if f"{name}.dt_bias" in missing:
                dt = torch.exp(
                    torch.empty_like(module.dt_bias).uniform_(0, 1)
                    * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
                ).clamp(min=1e-4)
                module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        elif isinstance(module, GatedDeltaNet):
            # fla's own scheme, which is NOT KDA's:
            #   A_log ~ log(uniform(0, 16))   -- KDA uses uniform(1, 16)
            #   dt_bias is per HEAD [20]      -- KDA's is per CHANNEL [2560]
            if f"{name}.A_log" in missing:
                module.A_log.copy_(
                    torch.empty_like(module.A_log).uniform_(0, 16).log())
            if f"{name}.dt_bias" in missing:
                dt = torch.exp(
                    torch.empty_like(module.dt_bias).uniform_(0, 1)
                    * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
                ).clamp(min=1e-4)
                module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
```

The loop below it needs no change: `normal(0, 0.02)` for 2-D weights, zeros for biases, ones for 1-D norms, with `.endswith((".A_log", ".dt_bias"))` excluded — all of which is already correct for GDN's thirteen tensors.

📌 **`transfer_attention_into_kda` needs no change at all.** It iterates `cfg.kda_layers` and reaches `student.model.layers[idx].self_attn.attn`, and the GDN adapter names its inner module `.attn` with `q/k/v/o_proj` all 2560×2560. ⚠️ Its *name* is now a misnomer in the same way `kda_layers` is; leave it.

### C-7. `finalize_checkpoint.py` — the marker

```python
    "AFMGDNLinearAttention": "Track C -- the GDN adapter is present",
```

⚠️ **Do not add this until the GDN student exists**, or every KDA checkpoint fails the check. The cleanest form is a per-mechanism marker set keyed off `config.json`'s `mechanism` field; the simple form is to add it when the first GDN checkpoint is built and accept that KDA checkpoints then need the KDA-only list.

---

## 6. `build_student.py` — six small changes, read 2026-09-23

⭐ **The existing four `VARIANTS` rows are not touched.** `spec.get("mechanism", "kda")` means every KDA build is provably identical to before — the same dict, the same constant, the same assertion.

**a. Beside `KDA_DELTA`:**

```python
KDA_DELTA = 11_883_668        # KDA layer (27,612,308) - Arcee attention layer (15,728,640)
GDN_DELTA = 17_172_648        # GDN layer (32,901,288) - Arcee attention layer (15,728,640)
GATE_PARAMS = 6_553_600       # gate_proj, 2560 x 2560

# Mechanism -> (per-converted-layer delta, adapter class name).  Adding a third
# linear-attention mechanism later means adding one row here and one adapter.
LINEAR_DELTA = {"kda": KDA_DELTA, "gdn": GDN_DELTA}
LINEAR_CLASS = {"kda": "AFMKDALinearAttention", "gdn": "AFMGDNLinearAttention"}
```

**b. Three new `VARIANTS` rows — same layouts, different mechanism:**

```python
    # ---- Track C: the GDN sibling.  Layout is orthogonal to mechanism, so
    # ---- these reuse their KDA counterparts' full_attn lists EXACTLY.
    "gdn_hybrid31": dict(
        dirname="afm_gdn_hybrid31_init",
        mechanism="gdn",
        full_attn=list(range(3, N_LAYERS, 4)),          # 3,7,11,...,35
        use_rope=False, use_attn_gate=True,
    ),
    "gdn_only": dict(
        dirname="afm_gdn_only_init",
        mechanism="gdn",
        full_attn=[],
        use_rope=False, use_attn_gate=True,
    ),
    "gdn_flp": dict(
        dirname="afm_gdn_flp_init",
        mechanism="gdn",
        full_attn=[0, 1, 2, 3, 11, 17, 23, 29, 35],     # OUR spacing -- see the flp warning
        use_rope=False, use_attn_gate=True,
    ),
```

**c. In `main()`, right after `spec = VARIANTS[args.variant]`:**

```python
    mechanism = spec.get("mechanism", "kda")
```

**d. ⚠️ The layout assertion must cover the GDN twin, or it silently stops applying:**

```python
    if args.variant in ("hybrid31", "gdn_hybrid31") and args.full_attn is None:
        assert kda_layers == build_kda_layers(N_LAYERS, 4), "hybrid31 layout drifted"
```

**e. The config call and the two assertions that use the mechanism:**

```python
    cfg = AFMKDAConfig.from_pretrained(
        args.teacher,
        kda_layers=kda_layers,                 # 0-BASED
        mechanism=mechanism,                   # <-- new
        use_rope=spec["use_rope"],
        use_attn_gate=spec["use_attn_gate"],
    )
    assert cfg.mechanism == mechanism, cfg.mechanism
```

```python
    expected = (teacher_params + n_kda * LINEAR_DELTA[mechanism]
                + (n_full * GATE_PARAMS if spec["use_attn_gate"] else 0))
```

```python
    kinds = [type(l.self_attn).__name__ for l in student.model.layers]
    linear_cls = LINEAR_CLASS[mechanism]
    assert kinds.count(linear_cls) == n_kda, kinds
    print(f"layer types: {kinds.count(linear_cls)} {mechanism.upper()}, "
          f"{kinds.count('AFMKDAAttention')} attention")
```

**f. `build_info.json` must record which mechanism was built**, or two checkpoints become indistinguishable from their metadata:

```python
        "mechanism": mechanism,
        "config": {
            "mechanism": mechanism,
            "kda_layers": kda_layers,
            ...
```

Also **replace** the existing early print, since the whole point is that two variants now differ by something the layer counts do not show:

```python
    print(f"variant       : {args.variant}   mechanism: {mechanism}")
```

⚠️ **It replaces the `print(f"variant       : {args.variant}")` line in the header block** — it does not go at the end of `main()`. At the end it prints after the save, and `--dry-run` returns before reaching it, so the one flag that decides which model is being built would be invisible in exactly the mode you use to check what is about to be built.

### ⚠️ And `build_student.sbatch` has the `--export` trap in it

```bash
VARIANT=flp                                    # <-- a PLAIN assignment
```

**That beats `--export`.** `sbatch --export=ALL,VARIANT=gdn_hybrid31 build_student.sbatch` would silently build `flp` — the trap is already written down in `startup_plan_and_pitfalls.md` and this file contains it. Fix:

```bash
VARIANT=${VARIANT:-hybrid31}
EXTRA=${EXTRA:-}                               # e.g. EXTRA="--no-transfer"

cd $WORK_LUKAS
python afm_kda/build_student.py --variant $VARIANT --force $EXTRA
```

---

## 6b. ✅ C0b — the pre-build structural check (login node, seconds, no GPU)

A build job is ~20 minutes and a queue slot. Almost everything it can catch is catchable first, by constructing **four layers at full width** with a tiny vocab and MLP — a few hundred MB instead of 10 GB, and no GPU.

It checks four things at once: the layer types (which catches an `AFMKDAModel.__init__` that lost its body — the symptom is every layer reading `ArceeAttention`), both per-layer parameter constants, whether `init_new_parameters` actually *reaches* each mechanism, and whether the model-level mechanism guard fires.

⭐ **The `init_new_parameters` probe is the interesting one.** `GatedDeltaNet.__init__` already initialises `A_log` and `dt_bias` correctly, so a freshly constructed module looks fine whether or not C-6 exists — the branch only matters after `from_pretrained` reports those keys missing. So the check **poisons them with NaN and sees whether they come back finite.** A constructor-time inspection would have passed and taught us nothing.

```python
BASE = dict(vocab_size=256, hidden_size=2560, num_hidden_layers=4,
            num_attention_heads=20, num_key_value_heads=4, head_dim=128,
            intermediate_size=512, kda_layers=[0, 1, 2],
            use_rope=False, use_attn_gate=True)

for mech in ("kda", "gdn"):
    m = AFMKDAForCausalLM(AFMKDAConfig(mechanism=mech, **BASE))
    mix = m.model.layers[0].self_attn.attn
    print([type(l.self_attn).__name__ for l in m.model.layers],
          sum(p.numel() for p in mix.parameters()),
          len(list(mix.parameters())))
    mix.A_log.data.fill_(float("nan")); mix.dt_bias.data.fill_(float("nan"))
    init_new_parameters(m, ["model.layers.0.self_attn.attn.A_log",
                            "model.layers.0.self_attn.attn.dt_bias"])
    print("re-initialised:", torch.isfinite(mix.A_log).all().item())

cfg = AFMKDAConfig(mechanism="kda", **BASE); cfg.mechanism = "gnd"
AFMKDAForCausalLM(cfg)          # must raise
```

**Result, 2026-09-23:**

| | layer types | per-layer params | tensors | `A_log` | `dt_bias` | init reached |
|---|---|---|---|---|---|---|
| `kda` | 3 × KDA + 1 attention ✅ | **27,612,308** ✅ | **16** ✅ | (20,) | **(2560,)** | ✅ |
| `gdn` | 3 × GDN + 1 attention ✅ | **32,901,288** ✅ | **13** ✅ | (20,) | **(20,)** | ❌ → C-6 was missing |

and the model-level guard rejected `"gnd"`. **The check earned its keep on its first run**: it found the one edit that had not been applied, and it confirmed both geometry constants without a queue slot.

⚠️ **Keep this as a reusable probe.** Any future mechanism — a third adapter, a changed head geometry — is one row in `EXPECT` away from being verified before a build.

---

## 7. C1 — the regression test

`claude/test_c1_gdn.py`. GDN has no "flags off = the teacher" property, so A3's `0.000e+00` has no analogue. The regression is built from four independently checkable things instead:

| # | check | what it catches |
|---|---|---|
| 1 | parameter count against the closed form | any geometry mistake |
| 2 | `loading_info` 360/108 from `build_info.json` | any naming or inventory mistake |
| 3 | forward **and** backward, gradient on every trainable tensor | a layer that runs but silently does not train — A5 caught real bugs this way |
| 4 | held-out CE at initialisation, beside the KDA student's 7.9412 | transfer or initialisation gone wrong |

1–3 are pass/fail with no invented tolerance. **4 is a sanity band, not a verdict** — both students are "the teacher's MLPs and embeddings plus random token mixing", so both should land near 8 and far below chance at `ln(128004) = 11.76`.

⚠️ **A GDN CE meaningfully BETTER than KDA's at initialisation would not be good news.** It would suggest the two are not comparably initialised, which is the confound the drop-in configuration exists to remove. Investigate before celebrating.

```bash
sed 's/test_b7c_generate/test_c1_gdn/g' $WORK_LUKAS/afm_kda/test_b7c_generate.sbatch \
  > $WORK_LUKAS/afm_kda/test_c1_gdn.sbatch
sbatch $WORK_LUKAS/afm_kda/test_c1_gdn.sbatch
```

---

## 8. Order of work

⭐ **No new files.** Every edit lands in the four files that already exist. §0 below says why.

The config edit stands alone and is testable in seconds. The four model edits do not — nothing constructs a GDN layer until all four are in — so they are one sitting.

```
[x] C0.  read kda.py and gdn.py side by side -- the mask guards, the init schemes
[x] C0.  geometry computed; five constants pinned for the assertion
[x] C0.  build_student.py read; VARIANTS structure known

--- step 1: the config, alone --------------------------------------------
[x] C-1. mechanism + gdn_expand_v + gdn_use_gate   configuration_afm_kda.py
[x] CHECK: both mechanisms construct, the DEFAULT is still "kda"
[x]        ...and the __post_init__ guard does NOT catch a kwarg -> C-4b

--- step 2: the model, one sitting ---------------------------------------
[x] C-2. import GatedDeltaNet                      modeling_afm_kda.py
[x] C-3. AFMGDNLinearAttention                     modeling_afm_kda.py
[x] C-4. decoder-layer dispatch, RAISING on unknown modeling_afm_kda.py
[x] C-4b. the mechanism guard in AFMKDAModel.__init__
[x] C-5. the stash loop   <-- SHARED CODE          modeling_afm_kda.py
[x] C-6. init_new_parameters GDN branch            modeling_afm_kda.py

--- C0b: the structural check, login node, seconds ----------------------
[x] layer types, both parameter constants, the guard      all OK
[ ] re-run after C-6 -> "init_new_parameters reached" must flip to OK for gdn

--- step 3: the builder --------------------------------------------------
[x] C-8. VARIANTS + LINEAR_DELTA + assertions      build_student.py
[x] C-9. VARIANT=${VARIANT:-hybrid31}              build_student.sbatch
[ ] move the mechanism print into the HEADER block, not the end of main()

--- step 4: commit BEFORE building ---------------------------------------
[x] committed at 06cf811, then 8f40624 (Track C verified), tree clean

--- step 5: two debug jobs, ran concurrently -----------------------------
[x] A3 (58447262) -> 0.000e+00 on all six checks
[x] build afm_gdn_hybrid31_init (58447265) -> 5,141,828,536, 360 / 108,
        27 GDN + 9 attention, 9.6 GiB, "dirty": false

--- step 6 ---------------------------------------------------------------
[x] C1 (58461599) -- all four checks; CE 7.9247 vs KDA's 7.9269
[x] cosmetic: the _is_hf_initialized diagnostic is commented out
[x] B7 on afm_gdn_hybrid31_init (58636620) -- ran; 10.6/10.1/5.5/6.9 ulp,
        27-29/32.  INCONCLUSIVE, two confounds -- see 8f
[x] CONTROL A: batch 1, zero pad -> 6.8 ulp.  Most of the magnitude is
        there before any pad token exists
[x] CONTROL B: untrained afm_kda_hybrid31_init -> 8.1/12.9/12.0/12.0, and
        its MOST-padded row is its CLEANEST.  Opposite ordering
[x] ✅ C-5 PASSES -- see 8g
[x] B7c on the re-synced output/r1_stage1_hybrid31 (58645587) -- fluent on
        all 3 rows, use_cache True, most-padded row best.  C-5 covered
        through generate() at last.  CONFIRM the run used sdpa
[x] C-7. finalize_checkpoint marker added -- 6/6 on both trained dirs
[x] B6 on afm_kda_only_init -- clean; fp32 1.439e-02, argmax 32/32,
        growth 8.1x vs the hybrid's 9.1x.  BUT the fallbacks are inert,
        not validated -- see 8c
[x] B7 on afm_kda_only_init (58635894) -- NO padding ordering at all
        (unpadded row 2nd worst).  The all-linear mask path behaves
[x] B6e at 128 steps (58645600) -- the curve is FLAT, not merely bounded.
        Last position 4x CLEANER than at 32 steps.  track_b.md rev 16
[x] build gdn_only -- 5,237,399,968, 468/144, 36 GDN + 0 attn, 9.8 GiB
[ ] rebuild the four KDA students at the Track C commit, so students/ runs
        ONE version of the code (sha256sum the weights first)
[ ] confirm the FLP layout against Arcee's config, THEN build gdn_flp
[ ] cosmetic: transfer_attention_into_kda() prints "into 27 KDA layers"
        on a GDN build
```

## 8b. ✅ The build — job 58447265, 2026-09-23

```
variant       : gdn_hybrid31   mechanism: gdn
KDA layers    : 27   full attention: 9 at [3, 7, 11, 15, 19, 23, 27, 31, 35]
teacher parameters: 4,619,184,640
missing keys: 360   unexpected keys: 108
transferred 108 tensors into 27 KDA layers
parameters: 5,141,828,536   expected 5,141,828,536   (5.141829 B)
layer types: 27 GDN, 9 attention
saved to .../students/afm_gdn_hybrid31_init  (9.6 GiB)
```

⭐ **`loading_info` 360/108 is the strongest single line here.** It re-verifies the whole initialisation inventory for free, the same way 441/108 did for KDA: **360 = 27 × 13 new tensors + 9 attention `gate_proj`**, and **108 = 27 × 4** teacher projections whose names no longer match because ours are one level deeper at `self_attn.attn.*`. Both land exactly, which means `from_pretrained` matched every name we expected it to and nothing we did not.

⭐ **And `"dirty": false` for the first time in this project.** The four KDA builds all record `true`; this one does not, so its recorded hash actually reconstructs the code that built it.

**A3 (58447262) passed all six checks**, `0.000e+00` on logits and all 37 hidden states — so adding the GDN import, the config fields, the dispatch and the model-level guard left the KDA path bit-identical. ⚠️ Note what that does **not** cover: A3's student has no linear layers, so the stash loop (C-5) is still untested. **B7c is what covers it.**

## 8c. ✅ C1 — job 58461599, 2026-09-25

| check | result |
|---|---|
| parameters | **5,141,828,536** ✅ · one GDN layer **32,901,288** ✅ |
| `loading_info` | **360 / 108** ✅ |
| forward + backward | loss 10.9248 (on random ids, so ≈ chance), **543/543 tensors with a gradient**, 0 non-finite ✅ |
| layers | 36 = 27 GDN + 9 attention, first attention at index 3 ✅ |

📌 **543 is a closed form too**, and it lands: `147` non-mixing tensors (embed + lm_head + final norm + 72 layernorms + 72 MLP) `+ 9 × 5` attention `+ 27 × 13` GDN `= 543`. *(The KDA student's equivalent is 147 + 45 + 27×16 = **624**, the number Track A records.)* Everything is trainable here because `frozen_res` is applied by DistillKit at training time, not at build time.

### ⭐⭐ The result: the two mechanisms start from the same place

| model | CE at initialisation | perplexity |
|---|---|---|
| **GDN student, untrained** | **7.9247** | 2,764.8 |
| **KDA student, untrained** | **7.9269** | 2,770.9 |
| teacher | 2.3030 | 10.0 |

**The two differ by 0.0022 nats.** Track A already established that "the untrained hybrid" is a *distribution, not a point* — A5b read 7.9412, R0 read 7.918, `test_r1_ce` read 7.9274, a spread of ~0.023 nats attributable to the random draw for the ★ parameters. **0.0022 is an order of magnitude inside that noise.**

⭐ **So the drop-in did its job.** Neither mechanism enjoys an initialisation advantage; both begin as "the teacher's MLPs and embeddings plus randomly-initialised token mixing", equally far from the teacher, and **any difference measured after training is attributable to the mechanism rather than to the starting point.** That is the single property the whole `expand_v=1, use_gate=True, 20×128` choice was made to secure, and it is now measured rather than argued.

⚠️ **Do not read GDN's 7.9247 as "slightly better".** It is 0.0022 against a 0.023 noise floor — a tenth of the scale at which this quantity is reproducible. The script's own warning is the right frame: a *meaningfully* better GDN number would have been evidence of a confound, not of quality.

### ⬜ What C1 made visible: the all-linear cache path had never been run

C1 loads `afm_gdn_hybrid31_init`, which has nine attention layers, so `_first_attn_layer_idx()` returns 3 and `AFMKDACache` takes its ordinary path. **An `only` student has no attention layers at all**, so `attn_layer_idx is None` and three fallbacks engage that have never run in any test:

| | hybrid | `only` |
|---|---|---|
| `get_seq_length()` | `super().get_seq_length(3)` | `self._counted`, maintained by `note_tokens` |
| `get_mask_sizes()` | `super().get_mask_sizes(q, 3)` | `(query_length, 0)` — a value nobody consumes |
| `note_tokens()` | no-op | **the only thing tracking position** |

⚠️ **This is not a GDN problem — it is untested for KDA too**, and `afm_kda_only_init` has existed since Track A. Arcee's own numbers put `kda_only` ahead of the hybrid on MMLU and ARC-C, and B9a puts it at 48.9× smaller state against 3.77×, so an `only` variant is the most likely second training run in the project — and **its evaluation would be the first thing ever to exercise this path.**

### ✅ …and the run that followed, 2026-09-25 — with a correction to what it was asked to prove

`test_b6_cache.sbatch` on `afm_kda_only_init` came back clean. Full table and reasoning in `claude/track_b.md` §4; the headline is that **fp32 reads 1.439e-02 against the hybrid's 8.021e-03, argmax 32/32, and the growth with position is 8.1× where the hybrid reads 9.1×** — so removing all nine attention anchors did *not* make the drift worse, which is the one thing a 36-layer unanchored stack could plausibly have broken.

⚠️ **The script's own printed verdict is stale and should not be used.** It asks for "~1e-5 or below"; that threshold was written before the TF32 finding, and the *validated* hybrid would fail it. Read the number against the hybrid, not against the script.

⚠️⚠️ **And the run proves less about the fallbacks than I claimed when proposing it.** In a model with no attention layers, **nothing consumes their answer**: `get_seq_length()` feeds `cache_position` → `position_ids` → RoPE, which no linear layer reads, and `get_mask_sizes()` feeds a causal mask that only an attention layer would consume — and the B6 family passes no mask at all, so it is never called. **`_counted` could be permanently zero and this test would still pass.** The fallbacks are **inert**, not **validated**. That is reassuring (inert code cannot be wrong in a way that matters) but it is a different claim.

⬜ **What actually gates an `only` student is Defect 2 — the padding mask — and it is still untested.** `track_b.md` §3 already says why it matters more here: with all 36 layers linear, a mask that fails to arrive makes left-padded batched generation *entirely* wrong rather than partly, and left-padded batches are exactly what lm-eval sends. **The job is B7, not B6:**

```bash
sbatch --export=ALL,MODEL=$WORK_LUKAS/students/afm_kda_only_init \
       $WORK_LUKAS/afm_kda/test_b7_batched.sbatch
```

⚠️ **Check first that `test_b7_batched.py` reads `MODEL` from the environment** the way `test_b6d_precision.py` does — it was written against the hybrid and may have the path hardcoded. One `grep -n "MODEL\|os.environ" test_b7_batched.py` settles it, and if the path is fixed, the two-line change is cheaper than a wasted slot.

## 8e. ✅ `afm_gdn_only_init` — built 2026-09-25

```
variant       : gdn_only   mechanism: gdn
KDA layers    : 36   full attention: 0 at []
missing keys: 468   unexpected keys: 144
transferred 144 tensors into 36 KDA layers
parameters: 5,237,399,968   expected 5,237,399,968
layer types: 36 GDN, 0 attention
saved to .../students/afm_gdn_only_init  (9.8 GiB)
```

Every number matches §4's closed form. ⭐ **The `loading_info` pair is the informative line, and the `only` variant sharpens it:** the hybrid reads `360 = 27×13 + 9`, where the `+9` is the attention layers' `gate_proj`; this one reads `468 = 36×13 + 0`, because there are no attention layers to carry a gate. `use_attn_gate: True` in the header is therefore **inert here**, and the parameter assertion passing is what proves the `n_full × GATE_PARAMS` term went to zero rather than being quietly dropped.

⚠️ **The `dirty` warning is the benign case §8d describes**, and the user's reading is right: it is `bool(git status --porcelain)` over the whole tree, and `test_a3.py`/`test_c1_gdn.py` were edited after `06cf811`. The five `.py` files copied into the checkpoint are still the committed ones, so the model is honest — the real fingerprint is `sha1sum $STUDENT/*.py`. **Commit the test files so the next build records clean**, because the point of the flag is that you should not have to reconstruct why it fired.

⭐ `afm_gdn_only_init` is now the only checkpoint combining **the current code**, **GDN**, and **zero attention layers** — so it is the subject that eventually covers the all-linear fallbacks and the GDN arm at once. Test them one at a time first (B7 on `afm_kda_only_init` for the all-linear path, B7 on `afm_gdn_hybrid31_init` for C-5); a combined test that fails tells you less.

## 8d. Should `gdn_only` and `gdn_flp` be built now?

**`gdn_only`: yes.** It is the paired ablation for `kda_only`, which Arcee's own published numbers make the most valuable second training run; building it costs one debug job and ~9.8 GiB, and the code that builds it is committed and verified.

**`gdn_flp`: not yet.** The FLP layout in `build_student.py` is flagged in its own comment as *our* choice — the blog post never gives the remaining five indices — and `building_the_student.md` still lists "flp layout CONFIRMED against `AFM-4.5B-Base-KDA-FLP/config.json`" as an open item. ⚠️ **Building a second student on an unverified layout doubles the cost of finding out it was wrong**, and turns a directory you would be tempted to trust into one you have to remember not to. Confirm the layout against their published config first, then build both FLP variants together from the same corrected list.

📌 **Neither `test_a3.py` nor `build_student.py` needs the re-sync loop.** Both put `$WORK_LUKAS` on `sys.path` and import `afm_kda.*` **from the repo**, so they run the code you just edited. The re-sync exists for *checkpoints*, which carry their own frozen `.py` copies — and the GDN build writes fresh ones into a directory that does not exist yet, so there is no stale module cache to clear either.

📌 **On `"dirty": true` in the existing builds.** It is `bool(git status --porcelain)` over the *whole tree*, so editing `build_student.sbatch` sets it even though no model code changed — and the model code's real fingerprint is the `sha1sum $STUDENT/*.py` line in the provenance block, which is unaffected. So those four flags are benign. **Commit first anyway**: the flag is a blunt instrument precisely because it cannot tell you which file was dirty, and a clean one is worth more than an explained one.

⚠️ **Commit before the build, not after.** The four existing `build_info.json` files record `"dirty": true`, which makes their recorded hash a half-truth. `build_student.py` prints a warning when it happens; do not earn it again.

⚠️ **Do not write this code while an R1-class job is queued or running** — a queued job picks up whatever is in the checkpoint when it starts.

📌 **The GSM8K and IFEval full runs are safe.** They load `output/r1_stage1_hybrid31`, which carries its own frozen `.py` copies, so repo edits cannot reach them. ⚠️ **But do not re-sync that directory** until they finish.

⚠️ **Do not write this code while an R1-class job is queued or running** — a queued job picks up whatever is in the checkpoint when it starts.

⚠️ **The GSM8K and IFEval full runs are in flight.** They load `output/r1_stage1_hybrid31`, which carries its own frozen `.py` copies, so repo edits cannot disturb them — but do not re-sync those directories until they finish.

---

## 8f. 🔶 B7 on two new students — measured 2026-09-25, *not interpretable until §8g*

Two runs, both `IMPL=sdpa`, both against the teacher control in the same process.

| | row 0 · pad 180 | row 1 · pad 120 | row 2 · pad 60 | row 3 · pad 0 | argmax |
|---|---|---|---|---|---|
| **teacher** (control) | 1.8 ulp | 1.5 | 1.2 | 1.5 | 31–32/32 |
| hybrid31 **trained** (R1, recorded in `track_b.md`) | 1.5 | 2.5 | 1.1 | 1.5 | 31–32/32 |
| `afm_kda_only_init` **untrained** (58635894) | **7.5** | **9.3** | **5.8** | **8.5** | 29–30/32 |
| `afm_gdn_hybrid31_init` **untrained** (58636620) | **10.6** | **10.1** | **5.5** | **6.9** | 27–29/32 |

⚠️ **Neither run meets the script's failure criterion** — "orders of magnitude worse" — and neither matches the trained hybrid either. **Two confounds have to come out before these numbers mean anything**, and each costs one debug job.

### ⚠️⚠️ Confound 1: in B7's default `LENS`, padding and length are perfectly anti-correlated

`LENS=120,180,240,300` padded to width 300 gives `pad = 300 − len` **exactly**. So "the most-padded row" and "the shortest row" are the same row, and the ordering check the script describes cannot separate *pad contamination* from *any length-dependent arithmetic difference*. And there are such differences by construction: with a mask present fla unpads into one flattened varlen sequence of 840 tokens and chunks it there, so a row's chunk alignment inside the batch differs from its standalone run, and differs per row.

**The control removes the correlation**, same script, one variable:

```bash
sbatch --export=ALL,MODEL=$WORK_LUKAS/students/afm_gdn_hybrid31_init,IMPL=sdpa,LENS=120,120,120,120 \
       $WORK_LUKAS/afm_kda/test_b7_batched.sbatch
```

Four rows of length 120, **zero padding on any of them**, same batch of 4. If the len-120 rows still read ~10 ulp, padding is not what the gradient tracks. If they fall to 1–2 ulp, it is.

### ⚠️ Confound 2: both students are **untrained**, and the trained hybrid is the only B7 baseline on record

An untrained student sits at CE 7.92, perplexity 2,765 — a nearly flat predictive distribution, where near-ties are everywhere and small perturbations flip the argmax. B6c already established that on this model. Comparing 27/32 on an untrained student against 31/32 on a trained one is not like for like, and the 4–6× ulp inflation may simply be what thirty-odd *randomly initialised* recurrent layers do to a perturbation.

**The control is the untrained KDA hybrid** — same layout, same training state, different mechanism:

```bash
sbatch --export=ALL,MODEL=$WORK_LUKAS/students/afm_kda_hybrid31_init,IMPL=sdpa \
       $WORK_LUKAS/afm_kda/test_b7_batched.sbatch
```

📌 It runs the pre-Track-C code, which is **fine for this purpose**: the KDA arm of the C-5 stash is byte-identical between the two versions, so the only thing that differs from run 58636620 is the mechanism.

### ⭐ What the two runs already say

**`kda_only` shows no padding ordering at all** — 7.5 / 9.3 / 5.8 / **8.5**, with the *unpadded* row second-worst and the worst at pad 120. In a model where all thirty-six layers are linear, `track_b.md` §3 says Defect 2 should bite *hardest*; a mask that was not arriving would put row 3 cleanest and row 0 worst. It does the opposite. **That is the all-linear padding path behaving**, and it is more than B6 could show, since B6 passes no mask at all.

**The GDN hybrid does lean the wrong way** — 10.6 / 10.1 / 5.5 / 6.9, the two most-padded rows worst by ~1.7×. That is the direction a C-5 failure would produce, and it is also exactly what confound 1 predicts, which is why it has to be removed rather than argued about.

📌 **Cosmetic, but read it before doubting which model ran:** the section header prints `STUDENT hybrid31` in both runs — it is a hardcoded string, not derived from `MODEL`. The evidence that different models loaded is that the student rows differ while the teacher rows are identical to the digit. Worth changing to print the path.

## 8g. ✅ §8f resolved — both controls came back, and **C-5 passes**

| student | state | pad 180 | pad 120 | pad 60 | pad 0 | argmax |
|---|---|---|---|---|---|---|
| teacher *(control)* | — | 1.8 ulp | 1.5 | 1.2 | 1.5 | 31–32/32 |
| hybrid31 | **trained** | 1.5 | 2.5 | 1.1 | 1.5 | 31–32/32 |
| `kda_only` | untrained | 7.5 | **9.3** | 5.8 | **8.5** | 29–30/32 |
| `gdn_hybrid31` | untrained | **10.6** | 10.1 | 5.5 | 6.9 | 27–29/32 |
| **`kda_hybrid31`** ← control B | untrained | **8.1** ← *best row* | 12.9 | 12.0 | **12.0** | 28–29/32 |
| **`gdn_hybrid31`, batch 1, zero pad** ← control A | untrained | — | — | — | **6.8** | 28/32 |

### ⭐⭐ Control A: most of the magnitude exists before any pad token does

**6.8 ulp, one row, zero padding, batch of one.** The GDN hybrid's worst padded row was 10.6. So **padding and batching together account for roughly 1.6×, and the other 6.8 is there with no pad token anywhere in the tensor.**

📌 *The likely mechanism, stated as inference rather than measurement:* an all-ones 2-D mask is not `is_segment_mask`, so the stash hands fla a mask of ones; fla runs `get_unpad_data` and takes its **varlen** path, while the unpadded reference (mask `None`) takes the **dense** path. Different kernel configuration, different accumulation order, a few ulp — the same class of difference as `chunk` against `fused_recurrent`. The teacher shows the same floor at 1.8 ulp on the identical row.

### ⭐⭐ Control B: the untrained **KDA** hybrid is worse, with the opposite ordering

`afm_kda_hybrid31_init` reads **8.1 / 12.9 / 12.0 / 12.0**, and its **most-padded row is its cleanest** while its **unpadded row is tied worst** — exactly the inversion B7c was written to look for, and the exact opposite of the GDN student's mild gradient.

⭐⭐ **Pad contamination does not produce opposite orderings in two models running the same stash loop.** The GDN student's 1.7× lean was four-row noise. And the GDN student is **better** than the KDA one on every summary statistic — 5.5–10.6 ulp against 8.1–12.9 — so there is no mechanism effect to explain either.

### The verdict

✅ **C-5's GDN arm works.** The padding mask reaches `AFMGDNLinearAttention`, and the tuple-membership edit did what it was written to do.
✅ **The all-linear padding path works** — `kda_only` shows no padding ordering at all.
⭐ **The untrained B7 baseline is 6–13 ulp and 27–30/32**, against a trained student's 1–2.5 and 31–32/32. ⚠️ **B7's published pass band was measured on a trained model and does not transfer.** Argmax agreement is a function of how peaked the distribution is; an untrained student sits at perplexity 2,765, where near-ties are everywhere. That band is now on record in `track_b.md` §4 so the next person does not re-derive it.

⚠️ **`LENS=120,120,120,120` silently collapsed to a batch of one** — the script builds its batch from *distinct* lengths, so the intended four-equal-rows test never ran. It answered the question anyway, and arguably better: batch 1 with no padding is the cleanest floor available. **Read the `batch N | lengths [...]` header before trusting what was tested.**

## 8h. ✅ The last two Track C tests — 2026-09-25

**B7c, job 58645587**, on the re-synced `output/r1_stage1_hybrid31` (6/6 markers, all five `.py` hashes matching the repo). Three prompts, left-padded to width 17 with pads `[11, 9, 0]`, 48 greedy tokens through `generate()`.

✅ Fluent, on-topic English on all three rows. `config.use_cache` and `generation_config.use_cache` both print **True**. ⭐ **The row with 11 of 17 columns padded is again the best output** — the same ordering B7c found the first time, and the specific failure this test exists to catch. 📌 The teacher looped on that row ("Paris is the capital of France" six times) and the student did not; `track_b.md` §7 already says not to read that as a quality signal. 📌 One small factual slip in the student's row 0 ("the most populous city in the European Union") — this test reads for **coherence**, not facts, and a base model at 300 M distillation tokens is entitled to it.

⚠️ **One thing to confirm before calling this C-5 coverage.** Under `flash_attention_2` the mask never becomes 4-D and the stash is never exercised, so only an **sdpa** run counts. The B7c output does not print the implementation:

```bash
grep -n "IMPL\|attn_implementation" $WORK_LUKAS/afm_kda/test_b7c_generate.py
```

It was `sed`-derived from `test_b7_batched.py`, whose default is sdpa, so this is almost certainly fine — but "almost certainly" is what §9's first trap is about.

**B6e at 128 steps, job 58645600** — full table and the retired caveat in `track_b.md` §4. Headline: the fp32 error curve is **flat**, the final position is **four times cleaner** than at 32 steps, argmax is **128/128**, and a linear extrapolation written down beforehand was wrong in the safe direction. Long generations carry no accumulating-drift risk, which is what IFEval needed.

## 9. Traps this track has produced, or inherited

- **⭐⭐ Two models with opposite orderings cannot both be contaminated.** The GDN hybrid's most-padded rows were its worst and the KDA hybrid's were its *best*, on the same stash loop. That single comparison retired a question no tolerance could have settled.
- **⚠️ An env var can be silently reinterpreted.** `LENS=120,120,120,120` became `batch 1 | lengths [120]` because the script builds from distinct lengths. **Read the header line that says what actually ran.**
- **⚠️⚠️ A test can vary two things at once and look like it varies one.** B7's default `LENS=120,180,240,300` padded to 300 makes `pad = 300 − len`, so padding and sequence length are perfectly anti-correlated and "the most-padded row is worst" is indistinguishable from "the shortest row is worst". **Check the design for collinearity before reading an ordering.**
- **⚠️ A baseline measured on a trained model is not a baseline for an untrained one.** Argmax agreement is a function of how peaked the distribution is, and an untrained student at perplexity 2,765 produces near-ties everywhere. Compare untrained to untrained.
- **⚠️ A hardcoded label in a test's output survives every `--export`.** `test_b7_batched.py` prints `STUDENT hybrid31` whatever `MODEL` points at. Print the path, or trust only the numbers.
- **⚠️ A `@strict` config validates the FILE, not the CALL.** `__post_init__` runs before transformers applies `from_pretrained` kwargs, so `AFMKDAConfig(mechanism="gnd")` raises while `AFMKDAConfig.from_pretrained(T, mechanism="gnd")` does not. **Any invariant that must hold belongs where the value is used**, not in the config.
- **⚠️ A dispatch with an `else` fallback turns a typo into a mislabelled model.** `else: KDA` would build a KDA student carrying a config that claims otherwise — running, training and benchmarking perfectly, wrong forever. Raise instead.
- **⚠️ `...` in a pasted code excerpt is valid Python.** An `__init__` that ends at `super().__init__(config)` + `Ellipsis` silently keeps the parent's layers — no KDA, no GDN, no error. Never write `...` for "rest unchanged"; show the whole method.
- **⭐ To test an initialisation path, poison it first.** `GatedDeltaNet.__init__` already sets `A_log` and `dt_bias` correctly, so inspecting a fresh module proves nothing about whether `init_new_parameters` reaches it. Filling them with NaN and checking they come back finite is what found the missing branch.
- **⭐ Four layers at full width cost a few hundred MB and catch most of what a build catches.** Geometry, layer types, init reachability and guards are all verifiable on a login node in seconds. Spend the queue slot on what only a real build can test: `from_pretrained` name matching, i.e. `loading_info`.
- **⚠️ A class-name string comparison is a silent extension point.** `AFMKDAModel.forward` decides who gets the padding mask by `type(...).__name__ == "AFMKDALinearAttention"`. A new adapter is simply not seen, and the failure is Track B's defect 2 all over again — wrong output, no error.
- **⚠️ Do not assume one fla layer's quirks carry to its sibling.** KDA overwrites a supplied `cu_seqlens`; GDN, same release, raises `NameError` instead. The same guard prevents both, for different reasons.
- **⚠️ Initialisation schemes differ in distribution *and* shape.** GDN's `A_log` is `log(uniform(0,16))` against KDA's `uniform(1,16)`, and its `dt_bias` is `[20]` against KDA's `[2560]`. Copying KDA's branch would be wrong twice.
- **⚠️ A field name is not worth breaking a checkpoint over.** `kda_layers` means "which layers are linear"; renaming it would invalidate `output/r1_stage1_hybrid31`'s `config.json`. Document the misnomer instead.
- **⭐ Share the geometry fields deliberately.** One set of head/dim/conv values makes the drop-in property structural — the two students *cannot* be given different geometry by accident, so only the studied differences survive.
- **⚠️ Duplicate rather than refactor when the original is load-bearing.** The mask/cache preamble is ten lines that took a week to get right and is what the trained checkpoints run. Two copies with a comment beat one shared base and a re-validation debt.
- **⚠️ Computed constants are a test, not a fact.** Five numbers in §4 are arithmetic. The build assertion is what turns them into measurements — and if it fails, the arithmetic is what is wrong.
- **⚠️ GDN exercises a code path KDA never does.** `_use_fused_qkv_conv` fires under exactly our training conditions.
- **⚠️ `use_qk_l2norm_in_kernel=True` in GDN too** — the A5b caveat carries: transferred q/k magnitude is discarded, only direction survives.
- **⚠️ GDN switches kernel at `q_len <= 64` exactly as KDA does** — so B6d's finding that cached and uncached paths run *different programs* applies unchanged, and any GDN equivalence test must be teacher-forced rather than free-running.

---

## Sources

- `gdn_installed_0.5.2.py` — `fla/layers/gated_deltanet.py` from the installed 0.5.2, **authoritative**
- `kda_installed_0.5.2.py` — the KDA counterpart, **authoritative**
- `configuration_afm_kda.py`, `modeling_afm_kda.py` — read at `d7cd0a0`, 2026-09-23
- `claude/building_the_student.md` — Track A, and the initialisation/transfer machinery this reuses
- `claude/track_b.md` — the cache bridge and both B7 fixes, all inherited
- `KDA_Kimi.pdf` — the fine-grained-gating claim this track tests
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)