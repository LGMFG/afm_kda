"""A3 -- sanity check (a): an all-attention student must equal the teacher.

kda_layers=[] and the flags at their defaults (use_rope=True, use_attn_gate=False)
means the student is structurally identical to AFM-4.5B-Base.  Loading the
teacher's weights must therefore reproduce the teacher's logits exactly, up to
bf16 rounding.  Anything else is a bug in our plumbing, not in KDA.
"""
import os, sys, torch
WORK = os.environ["WORK_LUKAS"]
sys.path.insert(0, WORK)                       # so `import afm_kda` resolves

from transformers import AutoModelForCausalLM, AutoTokenizer
from afm_kda.configuration_afm_kda import AFMKDAConfig, build_kda_layers
from afm_kda.modeling_afm_kda import AFMKDAForCausalLM

TEACHER = "arcee-ai/AFM-4.5B-Base"
DTYPE, DEV, IMPL = torch.bfloat16, "cuda", "sdpa"   # sdpa on both sides = like for like

# ---------------------------------------------------------------- teacher
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER, dtype=DTYPE, attn_implementation=IMPL
).to(DEV).eval()

# ---------------------------------------------------------------- student
# from_pretrained reads AFM's own config.json, so the student inherits the
# teacher's exact geometry.  kda_layers=[] means every layer stays ordinary
# attention -- that is what makes this a real check rather than a formality.
cfg = AFMKDAConfig.from_pretrained(TEACHER, kda_layers=[])
assert cfg.vocab_size == 128004, cfg.vocab_size   # must match the teacher, or KL breaks


student, info = AFMKDAForCausalLM.from_pretrained(
    TEACHER, config=cfg, dtype=DTYPE, attn_implementation=IMPL,
    output_loading_info=True,                   # <-- gives us the key lists
)
student = student.to(DEV).eval()

flagged = sum(1 for _, p in student.named_parameters()
              if getattr(p, "_is_hf_initialized", False))
total = sum(1 for _ in student.named_parameters())
print(f"\n_is_hf_initialized set on {flagged}/{total} parameters")

print("=== CHECK 1: weight loading ===")
for k in ("missing_keys", "unexpected_keys", "mismatched_keys"):
    print(f"{k}: {info.get(k)}")                # ALL THREE MUST BE EMPTY

print("\n=== CHECK 2: parameter count ===")
pt = sum(p.numel() for p in teacher.parameters())
ps = sum(p.numel() for p in student.parameters())
print(f"teacher {pt/1e9:.4f} B   student {ps/1e9:.4f} B   equal: {pt == ps}")

# ---------------------------------------------------------------- forward
tok = AutoTokenizer.from_pretrained(TEACHER)
ids = tok("The capital of France is Paris. " * 8, return_tensors="pt").input_ids.to(DEV)

with torch.no_grad():
    t_out = teacher(ids, output_hidden_states=True)
    s_out = student(ids, output_hidden_states=True)


print("\n=== CHECK 3: logits ===")
diff = (t_out.logits.float() - s_out.logits.float()).abs().max().item()
scale = t_out.logits.float().abs().max().item()
print(f"max |diff| = {diff:.3e}   (logit scale {scale:.1f})   relative {diff/scale:.2e}")

print("\n=== CHECK 4: hidden states ===")
# This is the one I am least sure about: transformers collects hidden states by
# hooking the classes named in _can_record_outputs.  If our subclass didn't wire
# that through, this prints 0 or None -- and DistillKit's hs_cosine would then be
# silently comparing nothing.
th = t_out.hidden_states
sh = s_out.hidden_states
print(f"teacher: {len(th) if th else th}   student: {len(sh) if sh else sh}   (expect 37 each)")
if th and sh and len(th) == len(sh):
    worst = max((th[i].float() - sh[i].float()).abs().max().item() for i in range(len(th)))
    print(f"max |diff| over all 37 hidden states: {worst:.3e}")

print("\n=== CHECK 5: layer types ===")
kinds = [type(l.self_attn).__name__ for l in student.model.layers]
print(f"{kinds.count('AFMKDAAttention')} attention, "
      f"{kinds.count('AFMKDALinearAttention')} KDA   (expect 36 / 0)")
print("\nA3 PASSES if: all key lists empty, param counts equal, "
      "relative logit diff < 1e-2, and 37 hidden states on both sides.")

print("\n=== CHECK 6: where is the NaN? ===")
bad = [n for n, p in student.named_parameters() if not torch.isfinite(p).all()]
print(f"non-finite parameters: {len(bad)}  {bad[:5]}")
print("teacher logits finite:", torch.isfinite(t_out.logits).all().item())
print("student logits finite:", torch.isfinite(s_out.logits).all().item())
print("student last hidden finite:", torch.isfinite(s_out.hidden_states[-1]).all().item())
print("lm_head finite:", torch.isfinite(student.lm_head.weight).all().item())
d = (student.lm_head.weight.float() - teacher.lm_head.weight.float()).abs().max().item()
print(f"lm_head max |diff| vs teacher: {d:.3e}")
print("embed_tokens max |diff|:",
      (student.model.embed_tokens.weight.float()
       - teacher.model.embed_tokens.weight.float()).abs().max().item())