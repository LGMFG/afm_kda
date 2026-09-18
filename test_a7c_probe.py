import os, torch, transformers
from transformers import AutoModelForCausalLM
from distillkit.monkey_patch_packing import get_unpad_data as _fn

# --- instrument DistillKit's patch -------------------------------------
calls = []
def probe(mask):
    calls.append(tuple(mask.shape))
    return _fn(mask)

assert hasattr(transformers, "modeling_flash_attention_utils")
transformers.modeling_flash_attention_utils._get_unpad_data = probe

# --- load exactly as DistillKit does -----------------------------------
P = f"{os.environ['WORK_LUKAS']}/students/afm_kda_hybrid31_init"
torch.manual_seed(0)
m = AutoModelForCausalLM.from_pretrained(
        P, attn_implementation="flash_attention_2",
        trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
print("attn impl:", m.config._attn_implementation)

# --- see what one retained ATTENTION layer is handed --------------------
def pre_hook(mod, args, kwargs):
    am = kwargs.get("attention_mask")
    print("[ATTN L3] mask:", None if am is None else (tuple(am.shape), int(am.max())),
          "| cu_seq_lens_q:", kwargs.get("cu_seq_lens_q"),
          "| cu_seqlens:", kwargs.get("cu_seqlens"))
    return None

m.model.layers[3].self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)

# --- one forward with a packed row --------------------------------------
LA, LB = 301, 200
V = m.config.vocab_size
ids = torch.randint(0, V, (1, LA + LB)).cuda()
seg = torch.cat([torch.ones(1, LA), 2 * torch.ones(1, LB)], 1).long().cuda()

with torch.no_grad():
    m(input_ids=ids, attention_mask=seg)

print("get_unpad_data calls:", len(calls), calls[:3])