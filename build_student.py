"""Build the untrained KDA student and save it as a loadable HF model directory."""
import os, sys, shutil, torch
WORK = os.environ["WORK_LUKAS"]
sys.path.insert(0, WORK)
torch.manual_seed(0)

from transformers import AutoModelForCausalLM, AutoTokenizer
from afm_kda.configuration_afm_kda import AFMKDAConfig, build_kda_layers
from afm_kda.modeling_afm_kda import (
    AFMKDAForCausalLM, init_new_parameters, transfer_attention_into_kda,
)

TEACHER = "arcee-ai/AFM-4.5B-Base"
OUT = f"{WORK}/afm_kda_student_init"

teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16)

cfg = AFMKDAConfig.from_pretrained(
    TEACHER,
    kda_layers=build_kda_layers(36),     # 27 KDA layers, 3:1
    use_rope=False,                      # NoPE, as the reference does
    use_attn_gate=True,                  # sigmoid output gate
)
assert cfg.model_type == "afm_kda", cfg.model_type
assert cfg.vocab_size == 128004, cfg.vocab_size

student, info = AFMKDAForCausalLM.from_pretrained(
    TEACHER, config=cfg, dtype=torch.bfloat16, output_loading_info=True)
init_new_parameters(student, info.get("missing_keys", []))
transfer_attention_into_kda(student, teacher)

n = sum(p.numel() for p in student.parameters())
print(f"parameters: {n/1e9:.4f} B   (expect 4.9990)")
assert not [k for k, p in student.named_parameters() if not torch.isfinite(p).all()]

student.save_pretrained(OUT)
AutoTokenizer.from_pretrained(TEACHER).save_pretrained(OUT)

# The .py files must live IN the model directory -- that is what auto_map points at.
for f in ("configuration_afm_kda.py", "modeling_afm_kda.py",
          "configuration_arcee.py", "modeling_arcee.py", "__init__.py"):
    shutil.copy(f"{WORK}/afm_kda/{f}", OUT)
print(f"saved to {OUT}")