from huggingface_hub.dataclasses import strict

from .configuration_arcee import ArceeConfig


def build_kda_layers(num_hidden_layers: int = 36, period: int = 4) -> list[int]:
    """0-based indices of the KDA layers for a (period-1):1 hybrid.

    With 36 layers and period 4 this returns everything except
    3, 7, 11, ..., 35 -- full attention every 4th layer, including the last.
    Same pattern as arcee-ai/AFM-4.5B-Base-KDA-NoPE, whose config expresses it
    1-based ([4, 8, ..., 36]).  We use 0-based because that is what
    `self.layers[i]` uses.
    """
    full = set(range(period - 1, num_hidden_layers, period))
    return [i for i in range(num_hidden_layers) if i not in full]


@strict
class AFMKDAConfig(ArceeConfig):
    """AFM with optional KDA layers.

    All three knobs default to AFM's own behaviour, so an AFMKDAConfig built
    from AFM's config.json describes the unmodified teacher -- which is what
    makes sanity check A3 a real check.
    """

    model_type = "afm_kda"

    # Points at THIS directory's filenames.  The reference model's in-class
    # auto_map is stale and survives only because its config.json overrides it.
    auto_map = {
        "AutoConfig": "configuration_afm_kda.AFMKDAConfig",
        "AutoModelForCausalLM": "modeling_afm_kda.AFMKDAForCausalLM",
    }

    # transformers 5.x configs are @strict dataclasses: fields are ANNOTATED
    # CLASS ATTRIBUTES, not __init__ arguments.  Defining an __init__ here
    # fights the machinery -- that was the bug in the first draft.
    kda_layers: list[int] | None = None   # 0-BASED; None/[] => pure attention
    kda_num_heads: int = 20               # 20 x 128 = 2560 = hidden_size
    kda_head_dim: int = 128
    kda_conv_size: int = 4                # fla calls this conv_size, not short_conv_kernel_size
    use_rope: bool = True                 # False => NoPE, as the reference does
    use_attn_gate: bool = False           # True  => sigmoid output gate, as the reference does

    def __post_init__(self, **kwargs):
        if self.kda_layers is None:
            self.kda_layers = []
        super().__post_init__(**kwargs)
        # config.json says "arcee"; from_pretrained sets that as an instance
        # attribute and shadows ours.  Ours must win.
        self.model_type = type(self).model_type
        # auto_map as a CLASS attribute is never serialised: to_dict() copies
        # self.__dict__, which holds instance attributes only.  Promote it, or
        # the saved config.json has no pointer to our .py files and reloading
        # fails with "Transformers does not recognize this architecture".
        self.auto_map = dict(type(self).auto_map)

    def is_kda_layer(self, layer_idx: int) -> bool:
        """0-based.  The reference does `(layer_idx + 1) in kda_layers`; we don't."""
        return layer_idx in self.kda_layers


__all__ = ["AFMKDAConfig", "build_kda_layers"]