import torch
from torch import nn

import math

from .configuration_afm_kda import AFMKDAConfig
# HF's remote-code loader copies only ONE level of relative imports.  modeling_arcee
# imports .configuration_arcee, and that second level is never copied -- the reload
# dies with FileNotFoundError on configuration_arcee.py.  Importing it here makes it
# a FIRST-level dependency of this file, which does get copied.
from .configuration_arcee import ArceeConfig  # noqa: F401

from .modeling_arcee import (
    ALL_ATTENTION_FUNCTIONS,
    ArceeAttention,
    ArceeDecoderLayer,
    ArceeForCausalLM,
    ArceeModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

try:
    from fla.layers.kda import KimiDeltaAttention
except ImportError as e:
    raise ImportError("pip install -U 'flash-linear-attention[cuda]'") from e

try:
    from fla.layers.gated_deltanet import GatedDeltaNet
except ImportError as e:
    raise ImportError("pip install -U 'flash-linear-attention[cuda]'") from e

# --- B2': the two-cache bridge -------------------------------------------
from fla.models.utils import Cache as FlaCache
from transformers.cache_utils import DynamicCache


class AFMKDACache(DynamicCache):
    """A transformers cache for the attention layers, carrying an fla cache for KDA.

    B0 established that handing fla's Cache to the inherited ArceeModel.forward
    raises inside transformers' own wrapper on transformers 5.x.  So we never do:
    transformers only ever sees a DynamicCache subclass, and the recurrent state
    rides along as an attribute.

    Two problems, one object:

    1. LENGTH.  DynamicCache.get_seq_length() reports layer 0, which in a hybrid
       is a KDA layer that stores nothing -- so it returns 0 forever and
       generate() never trims input_ids.  Measured at B0: get_seq_length() -> 0
       beside __len__() -> 36.  Read an ATTENTION layer instead.
    2. TRANSPORT.  generate() hands the SAME cache object back every step, so an
       attribute on it is a correctly-scoped home for the recurrent state.  No
       kwargs plumbing, no globals, no lifetime questions.

    attn_layer_idx=None means there are no attention layers (the kda_only
    variant), in which case nothing in the DynamicCache is ever written and the
    length has to be counted here instead.
    """

    def __init__(self, *args, attn_layer_idx=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_layer_idx = attn_layer_idx
        self.fla = FlaCache()
        self._counted = 0          # only used when attn_layer_idx is None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # The caller's layer_idx is ignored on purpose: only an attention layer
        # knows how long the sequence is.
        if self.attn_layer_idx is None:
            return self._counted
        return super().get_seq_length(self.attn_layer_idx)
    
    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        # DynamicCache answers from self.layers[layer_idx].  Its hybrid rescue
        # (skip to the first attention layer) fires only when the linear layers
        # are registered as non-CacheLayerMixin entries -- transformers 5.8.1
        # maps "linear_attention" to LinearAttentionLayer for exactly that.
        # Ours are not registered at all, so lazy replication fills 0..N with
        # EMPTY DynamicLayers and layer 0 reports kv_length = query_length,
        # which broadcasts to stride 0 and makes SDPA reject the mask.
        if self.attn_layer_idx is None:
            return query_length, 0
        return super().get_mask_sizes(query_length, self.attn_layer_idx)
    
    def note_tokens(self, n: int) -> None:
        """Called once per forward by AFMKDAModel when there is no attention layer."""
        if self.attn_layer_idx is None:
            self._counted += int(n)


class AFMKDAAttention(ArceeAttention):
    """AFM attention with two optional modifications, both OFF by default.

    use_rope=False      -> NoPE: skip rotary embeddings entirely.
    use_attn_gate=True  -> a learned sigmoid gate on the attention output,
                           mirroring the output gate KDA has internally.

    With both at their defaults this is exactly ArceeAttention, which is what
    makes sanity check A3 ("all-attention student reproduces the teacher")
    a real check rather than a formality.
    """

    def __init__(self, config: AFMKDAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.use_rope = config.use_rope
        self.use_attn_gate = config.use_attn_gate
        if self.use_attn_gate:
            self.gate_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=False,
            )

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        # fla's spelling would reach the flash interface as an unknown argument.
        kwargs.pop("cu_seqlens", None)
        # Body copied from ArceeAttention.forward.  Only the two marked lines differ.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_rope:                                              # <-- CHANGE 1
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        # Torch's SDPA is refusing a 4-D causal mask whose last dimension isn't contiguous
        if attention_mask is not None:
            attention_mask = attention_mask.contiguous()

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if self.use_attn_gate:                                         # <-- CHANGE 2
            attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states))
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class AFMKDALinearAttention(nn.Module):
    """Adapts fla's KimiDeltaAttention to ArceeDecoderLayer's call contract.

    ArceeDecoderLayer does `hidden_states, _ = self.self_attn(...)`, so this
    must return exactly two values.  KDA returns three.
    """

    def __init__(self, config: AFMKDAConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # Every argument passed by name.  The reference splats its whole config
        # block in, so its `short_conv_kernel_size` lands in **kwargs and is
        # silently ignored -- it happens to work only because fla's default
        # conv_size is also 4.
        self.attn = KimiDeltaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.kda_num_heads,
            head_dim=config.kda_head_dim,
            conv_size=config.kda_conv_size,
            mode="chunk",          # fla asserts chunk mode during training
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states,
        position_embeddings=None,   # accepted and ignored: KDA carries position implicitly
        attention_mask=None,
        past_key_values=None,
        **kwargs,                   # absorbs position_ids, use_cache, cu_seqlens, ...
    ):
        # KDA asserts a 2-D [batch, seq_len] padding mask.  What arrives here is
        # whatever create_causal_mask produced: None or a 2-D mask under
        # flash_attention_2, a 4-D causal mask under sdpa/eager.  Drop anything
        # that is not 2-D rather than tripping the assert.

        # We prefer the explicit value and stop handing fla a mask it would re-derive from
        # so we check for cu_seqlens first.  If it is present, we are in DistillKit's packed format
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is not None:
            attention_mask = None
        elif attention_mask is None or attention_mask.dim() != 2:
            # super().forward() already turned the 2-D padding mask into a 4-D
            # causal bias; fla asserts on anything but 2-D and needs the real
            # padding mask for its unpad/repad path.
            attention_mask = getattr(self, "_padding_mask_2d", None)

        # ---- B4': take the recurrent cache off the carrier ---------------
        fla_cache = getattr(past_key_values, "fla", None)
        want_cache = bool(kwargs.get("use_cache")) and fla_cache is not None

        if (fla_cache is None and past_key_values is not None
                and past_key_values.get_seq_length() > 0):
            raise NotImplementedError(
                "AFMKDALinearAttention was asked to continue from a populated cache "
                "that carries no fla state. This would silently produce wrong "
                "output. Use AFMKDACache, or use_cache=False."
            )

        out, _, _ = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=fla_cache,
            use_cache=want_cache,
            cu_seqlens=cu_seqlens,
        )
        return out, None            # no attention weights exist for a recurrent layer


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


class AFMKDADecoderLayer(ArceeDecoderLayer):
    """Same layer, but self_attn is chosen per index.

    Inherits ArceeDecoderLayer's forward unchanged -- and, importantly, its
    GradientCheckpointingLayer base, whose __call__(*args, **kwargs) forwards
    kwargs, so cu_seqlens survives checkpointing.
    """

    def __init__(self, config: AFMKDAConfig, layer_idx: int):
        super().__init__(config, layer_idx)  # builds an ArceeAttention we then replace
        if config.is_kda_layer(layer_idx):
            if config.mechanism == "kda":
                self.self_attn = AFMKDALinearAttention(config, layer_idx)
            elif config.mechanism == "gdn":
                self.self_attn = AFMGDNLinearAttention(config, layer_idx)
            else:
                raise ValueError(f"unknown mechanism {config.mechanism!r}")
        else:
            self.self_attn = AFMKDAAttention(config, layer_idx)


class AFMKDAModel(ArceeModel):
    config: AFMKDAConfig
    config_class = AFMKDAConfig
    _no_split_modules = ["AFMKDADecoderLayer"]
    _is_stateful = True                       # B3': generate() must not re-run
                                              #      the prefix itself
    _can_record_outputs = {
        "hidden_states": AFMKDADecoderLayer,
        "attentions": AFMKDAAttention,
    }

    def __init__(self, config: AFMKDAConfig):
        # ⚠️ The __post_init__ check does NOT catch a mechanism passed as a
        # from_pretrained kwarg -- transformers applies those AFTER __post_init__.
        # This is the point where a wrong value would silently build the wrong model.
        if config.mechanism not in ("kda", "gdn"):
            raise ValueError(
                f"mechanism must be 'kda' or 'gdn', got {config.mechanism!r}")
        super().__init__(config)
        self.layers = nn.ModuleList(
            [AFMKDADecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.post_init()

    def _first_attn_layer_idx(self):
        """Index of the first softmax-attention layer, or None for kda_only.

        Derived from the MODULES, not from a config list, so hybrid31 (first at
        3) and flp (first at 0) both work with no special-casing, and a renamed
        config field cannot silently return the wrong answer.
        """
        for i, layer in enumerate(self.layers):
            if isinstance(layer.self_attn, AFMKDAAttention):
                return i
        return None

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # ---- existing segment-mask block, unchanged ----
        if is_segment_mask(attention_mask):
            cu = build_cu_seqlens(attention_mask)
            max_len = int((cu[1:] - cu[:-1]).max())
            kwargs["cu_seqlens"] = cu
            kwargs["cu_seq_lens_q"] = cu
            kwargs["cu_seq_lens_k"] = cu
            kwargs["max_length_q"] = max_len
            kwargs["max_length_k"] = max_len

        # ---- B3': install the carrier cache -----------------------------
        # Dead during training: gradient checkpointing forces use_cache=False,
        # and this model's config.use_cache is False as well (measured at B0).
        if kwargs.get("use_cache"):
            pkv = kwargs.get("past_key_values")
            if not isinstance(pkv, AFMKDACache):
                if pkv is None or (isinstance(pkv, DynamicCache)
                                   and pkv.get_seq_length() == 0):
                    pkv = AFMKDACache(attn_layer_idx=self._first_attn_layer_idx())
                    kwargs["past_key_values"] = pkv
                else:
                    raise NotImplementedError(
                        "AFMKDAModel was handed a populated cache it did not create "
                        f"({type(pkv).__name__}, {pkv.get_seq_length()} tokens). The "
                        "KDA layers have no state in it, so continuing would silently "
                        "produce wrong output. Start from an empty cache, or pass "
                        "use_cache=False."
                    )
            if input_ids is not None:
                pkv.note_tokens(input_ids.shape[1])     # no-op unless kda_only

        raw2d = attention_mask if (attention_mask is not None
                                   and attention_mask.dim() == 2) else None

        for layer in self.layers:
            if type(layer.self_attn).__name__ in (
                    "AFMKDALinearAttention", "AFMGDNLinearAttention"):
                layer.self_attn._padding_mask_2d = raw2d

        return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

class AFMKDAForCausalLM(ArceeForCausalLM):
    config_class = AFMKDAConfig
    _no_split_modules = ["AFMKDADecoderLayer"]
    _can_record_outputs = AFMKDAModel._can_record_outputs
    _is_stateful = True

    def __init__(self, config: AFMKDAConfig):
        super().__init__(config)
        self.model = AFMKDAModel(config)
        self.post_init()


# helper for KDA's cu_seqlens argument, which is not part of the HF contract and
# must be built from the segment_mask that create_causal_mask produces.
def build_cu_seqlens(segment_mask: torch.Tensor) -> torch.Tensor:
    """[1, T] mask of 1-based segment ids ([1,1,1,2,2,...]) -> int32 cu_seqlens.

    Mirrors functionary's get_max_seqlen_in_batch: count the tokens carrying
    each id, then take the cumulative sum with a leading 0.
    """
    row = segment_mask[0]
    lens = [n for i in range(1, int(row.max()) + 1)
            if (n := int((row == i).sum())) > 0]
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device=segment_mask.device)
    cu[1:] = torch.tensor(lens, dtype=torch.int32, device=segment_mask.device).cumsum(0)
    return cu

# Check if the attention mask is a segment mask.
def is_segment_mask(attention_mask) -> bool:
    """True when the packer wrote 1-based segment ids into attention_mask.

    An ordinary 0/1 padding mask has max() == 1 and is left alone, which is
    why A3 stays bit-identical.
    """
    return (
        attention_mask is not None
        and attention_mask.dim() == 2
        and attention_mask.shape[0] == 1      # per_device_train_batch_size: 1
        and int(attention_mask.max()) > 1
    )


@torch.no_grad()
def init_new_parameters(model, missing_keys):
    """Initialise exactly the parameters that were NOT in the checkpoint.

    Done explicitly rather than by overriding _init_weights, because that hook
    also fires for parameters transformers has already loaded -- which silently
    overwrote the teacher's weights (A3's logit diff went 0.0 -> 12.0).
    `missing_keys` comes from from_pretrained(..., output_loading_info=True),
    so this touches nothing else by construction.
    """
    std = model.config.initializer_range
    missing = set(missing_keys)

    # KDA's two special parameters need fla's own scheme, not a normal draw.
    # GDN's added
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

    for name, p in model.named_parameters():
        if name not in missing or name.endswith((".A_log", ".dt_bias")):
            continue
        if p.dim() >= 2:                 # Linear and Conv1d weights
            nn.init.normal_(p, mean=0.0, std=std)
        elif name.endswith(".bias"):
            nn.init.zeros_(p)
        else:                            # 1-D norm weights (o_norm)
            nn.init.ones_(p)


@torch.no_grad()
def transfer_attention_into_kda(student, teacher, verbose=True):
    """Copy the teacher's attention projections into the KDA layers.

    Two problems to solve:

    1. NAMES.  Our wrapper nests KDA as `self_attn.attn.*`, so the teacher's
       `self_attn.q_proj` matches nothing in a KDA layer -- not even the
       same-shaped q_proj and o_proj.  from_pretrained reports them as
       UNEXPECTED / MISSING respectively.

    2. SHAPES.  The teacher uses grouped-query attention: k_proj and v_proj
       output only num_key_value_heads * head_dim = 4 * 128 = 512, and those
       4 heads are shared across 20 query heads.  KDA has no sharing, so each
       KV head block must be repeated n_rep = 5 times.

       The repeat must match how attention itself expands them.  transformers'
       repeat_kv does `.expand(b, n_kv, n_rep, ...).reshape(b, n_kv*n_rep, ...)`,
       i.e. GROUP-MAJOR: kv head 0 repeated 5x, then kv head 1, and so on, so
       query head h uses kv head h // 5.  repeat_interleave on the head axis
       reproduces exactly that ordering.
    """
    cfg = student.config
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads      # 5
    hd = cfg.head_dim                                               # 128
    tsd = teacher.state_dict()
    moved = []

    for idx in cfg.kda_layers:
        src = f"model.layers.{idx}.self_attn"
        dst = student.model.layers[idx].self_attn.attn               # the KDA module

        # Same shape (2560 x 2560) -- a plain copy.
        for proj in ("q_proj", "o_proj"):
            w = tsd[f"{src}.{proj}.weight"]
            tgt = getattr(dst, proj).weight
            assert w.shape == tgt.shape, f"{proj}: {tuple(w.shape)} vs {tuple(tgt.shape)}"
            tgt.copy_(w.to(tgt.dtype, copy=True))
            moved.append(f"{src}.{proj}")

        # Narrow (512 x 2560) -> full width (2560 x 2560) via a 5x head repeat.
        for proj in ("k_proj", "v_proj"):
            w = tsd[f"{src}.{proj}.weight"]                          # [512, 2560]
            w = w.view(cfg.num_key_value_heads, hd, -1)              # [4, 128, 2560]
            w = w.repeat_interleave(n_rep, dim=0)                    # [20, 128, 2560]
            w = w.reshape(cfg.num_attention_heads * hd, -1)          # [2560, 2560]
            tgt = getattr(dst, proj).weight
            assert w.shape == tgt.shape, f"{proj}: {tuple(w.shape)} vs {tuple(tgt.shape)}"
            tgt.copy_(w.to(tgt.dtype, copy=True))
            moved.append(f"{src}.{proj} (repeat {n_rep}x)")

    if verbose:
        print(f"transferred {len(moved)} tensors into {len(cfg.kda_layers)} KDA layers")
    return moved

assert "forward" in AFMKDAModel.__dict__, \
    "AFMKDAModel.forward is not defined at class level -- check indentation"
assert issubclass(AFMKDACache, DynamicCache), \
    "AFMKDACache must be a DynamicCache: transformers must never see fla's Cache"

__all__ = ["AFMKDAConfig", "AFMKDAForCausalLM", "AFMKDAModel"]