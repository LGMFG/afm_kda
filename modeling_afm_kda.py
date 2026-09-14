import torch
from torch import nn

import math

from .configuration_afm_kda import AFMKDAConfig
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
        if attention_mask is not None and attention_mask.dim() != 2:
            attention_mask = None

        out, _, _ = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,   # fla's Cache is not transformers' Cache
            use_cache=False,        # training only; generation is separate work
            cu_seqlens=kwargs.get("cu_seqlens"),
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
            self.self_attn = AFMKDALinearAttention(config, layer_idx)
        else:
            self.self_attn = AFMKDAAttention(config, layer_idx)


class AFMKDAModel(ArceeModel):
    config: AFMKDAConfig          # transformers 5.x reads the annotation
    config_class = AFMKDAConfig   # kept for older call paths; harmless
    _no_split_modules = ["AFMKDADecoderLayer"]
    # CRITICAL: transformers collects hidden states by hooking the classes named
    # here (see @capture_outputs on ArceeModel.forward and _can_record_outputs on
    # ArceePreTrainedModel).  Leave this pointing at ArceeDecoderLayer and
    # output_hidden_states=True returns nothing -- which would silently zero out
    # DistillKit's hs_cosine loss with no error anywhere.
    _can_record_outputs = {
        "hidden_states": AFMKDADecoderLayer,
        "attentions": AFMKDAAttention,
    }

    def __init__(self, config: AFMKDAConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [AFMKDADecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.post_init()


class AFMKDAForCausalLM(ArceeForCausalLM):
    config_class = AFMKDAConfig
    _no_split_modules = ["AFMKDADecoderLayer"]
    _can_record_outputs = AFMKDAModel._can_record_outputs

    def __init__(self, config: AFMKDAConfig):
        super().__init__(config)
        self.model = AFMKDAModel(config)
        self.post_init()

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
    for name, module in model.named_modules():
        if not isinstance(module, KimiDeltaAttention):
            continue
        if f"{name}.A_log" in missing:
            module.A_log.copy_(torch.empty_like(module.A_log).uniform_(1, 16).log())
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


__all__ = ["AFMKDAConfig", "AFMKDAForCausalLM", "AFMKDAModel"]