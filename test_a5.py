"""A5b: verify the weight transfer, then measure whether it helps.

Rows differ ONLY in what was copied into the KDA layers.  Architecture,
seed, and evaluation tokens are identical throughout, so the differences
are attributable to the transfer and nothing else.
"""
import os, sys, gc, torch
WORK = os.environ["WORK_LUKAS"]
sys.path.insert(0, WORK)
torch.manual_seed(0)

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from afm_kda.configuration_afm_kda import AFMKDAConfig, build_kda_layers
from afm_kda.modeling_afm_kda import (
    AFMKDAForCausalLM, init_new_parameters, transfer_attention_into_kda,
)

TEACHER = "arcee-ai/AFM-4.5B-Base"
DTYPE, DEV, IMPL = torch.bfloat16, "cuda", "sdpa"
SEQ_LEN, N_SEQ = 2048, 8

tok = AutoTokenizer.from_pretrained(TEACHER)
ds = load_from_disk(f"{WORK}/datasets/fineweb_edu_3B_heldout")
buf = []
for row in ds:
    buf.extend(tok(row["text"]).input_ids)
    if len(buf) >= SEQ_LEN * N_SEQ:
        break
ids = torch.tensor(buf[: SEQ_LEN * N_SEQ]).view(N_SEQ, SEQ_LEN).to(DEV)

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER, dtype=DTYPE, attn_implementation=IMPL).to(DEV).eval()


@torch.no_grad()
def mean_ce(model):
    return sum(model(ids[i:i+1], labels=ids[i:i+1]).loss.float().item()
               for i in range(N_SEQ)) / N_SEQ


def build(kda_layers, transfer):
    cfg = AFMKDAConfig.from_pretrained(TEACHER, kda_layers=kda_layers,
                                       use_rope=False, use_attn_gate=True)
    m, info = AFMKDAForCausalLM.from_pretrained(
        TEACHER, config=cfg, dtype=DTYPE, attn_implementation=IMPL,
        output_loading_info=True)
    init_new_parameters(m, info.get("missing_keys", []))
    if transfer:
        transfer_attention_into_kda(m, teacher, verbose=False)
    return m.to(DEV).eval()


# ---------------------------------------------------------------- verification
print("=== VERIFY the transfer on one layer ===")
m = build([0], transfer=True)
t_attn = teacher.model.layers[0].self_attn
s_attn = m.model.layers[0].self_attn.attn
n_rep = m.config.num_attention_heads // m.config.num_key_value_heads
hd = m.config.head_dim

print("q_proj identical:",
      torch.equal(s_attn.q_proj.weight, t_attn.q_proj.weight.to(s_attn.q_proj.weight.dtype)))
print("o_proj identical:",
      torch.equal(s_attn.o_proj.weight, t_attn.o_proj.weight.to(s_attn.o_proj.weight.dtype)))

ok = True
for h in range(m.config.num_attention_heads):          # student head h <- teacher kv head h//5
    s_blk = s_attn.k_proj.weight[h*hd:(h+1)*hd]
    g = h // n_rep
    t_blk = t_attn.k_proj.weight[g*hd:(g+1)*hd].to(s_blk.dtype)
    if not torch.equal(s_blk, t_blk):
        ok = False
        print(f"  MISMATCH at head {h} (expected teacher kv head {g})")
print(f"k_proj head mapping correct for all {m.config.num_attention_heads} heads:", ok)
print("k_proj shape:", tuple(s_attn.k_proj.weight.shape), "(expect (2560, 2560))")
del m; gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------- ablation
ce_t = mean_ce(teacher)
print(f"\nTEACHER CE {ce_t:.4f}\n")
print(f"{'configuration':<38} {'CE':>8} {'vs teacher':>11}")
print("-" * 60)
KDA_ALL = build_kda_layers(36)
for name, layers, transfer in [
    ("1 KDA layer,  random init",   [0],     False),
    ("1 KDA layer,  transferred",   [0],     True),
    ("full hybrid,  random init",   KDA_ALL, False),
    ("full hybrid,  transferred",   KDA_ALL, True),
]:
    model = build(layers, transfer)
    ce = mean_ce(model)
    print(f"{name:<38} {ce:>8.4f} {ce - ce_t:>+11.4f}")
    del model; gc.collect(); torch.cuda.empty_cache()