"""A3b + A4 + A5: measure each architectural change, one at a time.

Every configuration is scored the same way -- mean cross-entropy on the SAME
sequences from the held-out split -- so the rows are directly comparable.
The baseline row must equal the teacher, which re-confirms A3.
"""
import os, sys, gc, torch
WORK = os.environ["WORK_LUKAS"]
sys.path.insert(0, WORK)

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from afm_kda.configuration_afm_kda import AFMKDAConfig, build_kda_layers
from afm_kda.modeling_afm_kda import AFMKDAForCausalLM, init_new_parameters

torch.manual_seed(0)

TEACHER = "arcee-ai/AFM-4.5B-Base"
DTYPE, DEV, IMPL = torch.bfloat16, "cuda", "sdpa"   # sdpa => a 4-D mask, which
SEQ_LEN, N_SEQ = 2048, 8                            # exercises the wrapper's drop path

tok = AutoTokenizer.from_pretrained(TEACHER)

# ---- one fixed batch, built once, so every row sees identical tokens --------
ds = load_from_disk(f"{WORK}/datasets/fineweb_edu_3B_heldout")
buf = []
for row in ds:
    buf.extend(tok(row["text"]).input_ids)
    if len(buf) >= SEQ_LEN * N_SEQ:
        break
ids = torch.tensor(buf[: SEQ_LEN * N_SEQ]).view(N_SEQ, SEQ_LEN).to(DEV)
print(f"eval batch: {N_SEQ} x {SEQ_LEN} tokens from the held-out split\n")


@torch.no_grad()
def mean_ce(model):
    """Mean cross-entropy, one sequence at a time to keep memory flat."""
    return sum(model(ids[i:i+1], labels=ids[i:i+1]).loss.float().item()
               for i in range(N_SEQ)) / N_SEQ


def backward_ok(model):
    """Does a gradient actually reach every trainable parameter?"""
    model.train()
    row = ids[:1, :512]                       # short, to keep activations small
    loss = model(row, labels=row).loss
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_grad = sum(1 for p in trainable if p.grad is not None)
    model.zero_grad(set_to_none=True)
    model.eval()
    return loss.item(), n_grad, len(trainable)


def build(**flags):
    cfg = AFMKDAConfig.from_pretrained(TEACHER, **flags)
    m, info = AFMKDAForCausalLM.from_pretrained(
        TEACHER, config=cfg, dtype=DTYPE, attn_implementation=IMPL,
        output_loading_info=True,
    )
    init_new_parameters(m, info.get("missing_keys", [])) # new line for adding KDA/gate params, if any
    return m.to(DEV).eval(), info

def nonfinite_params(model):
    """Names of parameters containing NaN or Inf -- i.e. never initialised."""
    return [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]


# ---- teacher first, then free it so the students have room -----------------
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER, dtype=DTYPE, attn_implementation=IMPL).to(DEV).eval()
ce_teacher = mean_ce(teacher)
n_teacher = sum(p.numel() for p in teacher.parameters())
print(f"TEACHER  CE {ce_teacher:.4f}   params {n_teacher/1e9:.4f} B\n")
del teacher; gc.collect(); torch.cuda.empty_cache()

KDA_ALL = build_kda_layers(36)
CONFIGS = [
    ("A3   baseline (= teacher)", dict(kda_layers=[]),                                   False),
    ("A3b  NoPE only",            dict(kda_layers=[], use_rope=False),                   False),
    ("A3b  attn gate only",       dict(kda_layers=[], use_attn_gate=True),               False),
    ("A4   1 KDA layer (idx 0)",  dict(kda_layers=[0]),                                  True),
    ("A5   full hybrid (27 KDA)", dict(kda_layers=KDA_ALL, use_rope=False,
                                       use_attn_gate=True),                              True),
]

print(f"{'configuration':<28} {'CE':>8} {'vs teacher':>11} {'params':>10}  notes")
print("-" * 90)
for name, flags, check_backward in CONFIGS:
    model, info = build(**flags)
    bad = nonfinite_params(model)
    if bad:
        notes.append(f"!! {len(bad)} non-finite params, first: {bad[0]}")
    ce = mean_ce(model)
    n = sum(p.numel() for p in model.parameters())
    notes = []
    if info.get("missing_keys"):
        notes.append(f"{len(info['missing_keys'])} new params (expected for KDA/gate)")
    if info.get("unexpected_keys"):
        # For rows WITH KDA layers this is expected: the wrapper nests KDA as
        # self_attn.attn.*, so the teacher's self_attn.q/k/v/o_proj in those
        # layers match nothing and are discarded.  For rows WITHOUT KDA layers
        # (baseline, NoPE, gate) it is a real bug.
        tag = "" if flags.get("kda_layers") else "!! "
        notes.append(f"{tag}{len(info['unexpected_keys'])} unexpected")
    if info.get("mismatched_keys"):
        notes.append(f"!! {len(info['mismatched_keys'])} MISMATCHED shapes")
    if check_backward:
        kda = next(l.self_attn for l in model.model.layers
                   if type(l.self_attn).__name__ == "AFMKDALinearAttention")
        notes.append(f"kda dtype {kda.attn.q_proj.weight.dtype}")
        bl, ng, nt = backward_ok(model)
        notes.append(f"backward loss {bl:.4f}, grads {ng}/{nt}")
    print(f"{name:<28} {ce:>8.4f} {ce - ce_teacher:>+11.4f} {n/1e9:>9.4f} B  "
          + "; ".join(notes))
    del model; gc.collect(); torch.cuda.empty_cache()