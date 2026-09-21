#!/usr/bin/env python
"""B6 -- is the cached path numerically the same as the uncached path?

TWO tests, and the first one is the real one.

  TEST 1  TEACHER-FORCED (deterministic).  Both paths are driven along the
          SAME fixed token sequence, so they compute the same thing and the
          expected answer is "equal to within bf16 rounding".  No argmax, no
          branching, no chaos.  THIS is the test of the cache.

  TEST 2  FREE-RUNNING GREEDY (chaotic).  Each path picks its own next token.
          ⚠️ Greedy decoding is a chaotic map: when the top two logits are
          within rounding distance, an arbitrarily small numerical difference
          flips the argmax and the two sequences diverge completely and
          permanently.  A divergence here is EXPECTED at near-ties and does
          not by itself indicate a bug -- which is why the top-2 gap is
          printed at the point of divergence.

  (This is A7's lesson again: when a test needs a tolerance you invented,
   redesign the test so the expected answer is exact.)
"""
import math
import os
import sys

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = os.environ["WORK_LUKAS"]
MODEL = os.environ.get("MODEL", f"{WORK}/output/r1_stage1_hybrid31")
N_NEW = int(os.environ.get("N_NEW", 128))
N_PROMPT = int(os.environ.get("N_PROMPT", 300))
N_RUNS = int(os.environ.get("N_RUNS", 3))
DTYPE, DEV, IMPL = torch.bfloat16, "cuda", "sdpa"
MB = 1024 ** 2


# Local copy on purpose: importing test_b9_cache_memory would RUN it.
def cache_bytes(obj, seen=None):
    if obj is None:
        return 0
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple, set)):
        return sum(cache_bytes(x, seen) for x in obj)
    if isinstance(obj, dict):
        return sum(cache_bytes(x, seen) for x in obj.values())
    d = getattr(obj, "__dict__", None)
    return sum(cache_bytes(x, seen) for x in d.values()) if d else 0


def ulp(x):
    """One unit in the last place of bfloat16 at magnitude x (8 significant bits)."""
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else float("nan")


tok = AutoTokenizer.from_pretrained("arcee-ai/AFM-4.5B-Base")
ds = load_from_disk(f"{WORK}/datasets/fineweb_edu_3B_heldout")

NEED = N_PROMPT + N_NEW
seqs, buf = [], []
for row in ds:
    buf.extend(tok(row["text"]).input_ids)
    while len(buf) >= NEED and len(seqs) < N_RUNS:
        seqs.append(torch.tensor(buf[:NEED], device=DEV).unsqueeze(0))
        buf = buf[NEED:]
    if len(seqs) == N_RUNS:
        break
assert len(seqs) == N_RUNS, "held-out split too small"

m = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=DTYPE, attn_implementation=IMPL,
    trust_remote_code=True).to(DEV).eval()

_mod = sys.modules[type(m).__module__]
assert hasattr(_mod, "AFMKDACache"), (
    f"{MODEL} is running PRE-B2' code ({_mod.__file__}). Re-sync "
    "modeling_afm_kda.py and clear $HF_HOME/modules."
)

n_kda = sum(1 for l in m.model.layers
            if type(l.self_attn).__name__ == "AFMKDALinearAttention")
expect_fla = n_kda * (20 * 128 * 128 * 4 + 3 * 2560 * 4 * 2) / MB
print(f"{type(m).__name__} | _is_stateful: {getattr(m.model, '_is_stateful', 'ABSENT')} "
      f"| {n_kda} KDA layers")
print(f"{N_RUNS} sequences | prompt {N_PROMPT} | {N_NEW} forced/generated tokens\n")


@torch.no_grad()
def forced(full):
    """TEST 1.  Identical inputs to both paths -- the expected answer is exact."""
    prompt, cont = full[:, :N_PROMPT], full[:, N_PROMPT:]

    # A: one pass over everything, no cache.  Prediction for cont[k] is at
    # position N_PROMPT-1+k.
    a = m(input_ids=full, use_cache=False).logits[0, N_PROMPT - 1: N_PROMPT - 1 + N_NEW]

    # B: prompt with cache, then the SAME tokens fed one at a time.
    out = m(input_ids=prompt, use_cache=True)
    pkv = out.past_key_values
    fla0 = cache_bytes(getattr(pkv, "fla", None)) / MB
    b = [out.logits[0, -1]]
    for k in range(N_NEW - 1):
        out = m(input_ids=cont[:, k: k + 1], past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        b.append(out.logits[0, -1])
    b = torch.stack(b)
    fla1 = cache_bytes(getattr(pkv, "fla", None)) / MB

    d = (a.float() - b.float()).abs()
    per_pos = d.max(dim=-1).values
    scale = a.float().abs().max().item()
    return dict(worst=d.max().item(), first=per_pos[0].item(),
                last=per_pos[-1].item(), scale=scale,
                fla0=fla0, fla1=fla1, seq=pkv.get_seq_length(),
                kv=(cache_bytes(pkv) - fla1 * MB) / MB)


@torch.no_grad()
def greedy(prompt):
    """TEST 2.  Each path chooses its own tokens -- chaotic by construction."""
    ids = prompt.clone()
    a_tok, a_gap = [], []
    for _ in range(N_NEW):
        lg = m(input_ids=ids, use_cache=False).logits[0, -1].float()
        top2 = lg.topk(2).values
        a_gap.append((top2[0] - top2[1]).item())
        nxt = lg.argmax()
        a_tok.append(nxt.item())
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)

    out = m(input_ids=prompt, use_cache=True)
    pkv = out.past_key_values
    b_tok = [out.logits[0, -1].argmax().item()]
    nxt = torch.tensor([[b_tok[0]]], device=DEV)
    for _ in range(N_NEW - 1):
        out = m(input_ids=nxt, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        b_tok.append(nxt.item())

    c_tok = m.generate(prompt, max_new_tokens=N_NEW, do_sample=False,
                       use_cache=True,
                       attention_mask=torch.ones_like(prompt),
                       pad_token_id=tok.eos_token_id)[0, N_PROMPT:].tolist()
    return a_tok, b_tok, c_tok, a_gap


ok = True
for i, full in enumerate(seqs):
    print(f"--- sequence {i}: {repr(tok.decode(full[0][:40]))[:100]}")

    f = forced(full)
    u = ulp(f["scale"])
    passed = f["worst"] <= 2 * u
    ok &= passed and f["fla0"] > 0 and abs(f["fla1"] - f["fla0"]) < 1e-6
    print(f"  TEST 1 teacher-forced   worst |Δlogit| over {N_NEW} positions: "
          f"{f['worst']:.3e}")
    print(f"           first position {f['first']:.3e} -> last {f['last']:.3e}   "
          f"(error must NOT grow)")
    print(f"           logit scale {f['scale']:.1f}, one bf16 ulp = {u:.3f}   "
          f"=> {'PASS' if passed else '*** FAIL ***'}")
    print(f"           fla state {f['fla0']:.2f} -> {f['fla1']:.2f} MB "
          f"(expect {expect_fla:.2f}, CONSTANT) | attn KV {f['kv']:.2f} MB | "
          f"get_seq_length {f['seq']}")

    a_tok, b_tok, c_tok, gap = greedy(full[:, :N_PROMPT])
    same_b, same_c = a_tok == b_tok, a_tok == c_tok
    print(f"  TEST 2 free greedy      A==B {same_b}   A==C {same_c}")
    if not same_b:
        k = next(j for j, (x, y) in enumerate(zip(a_tok, b_tok)) if x != y)
        print(f"           diverged at token {k}: top-2 logit gap there was "
              f"{gap[k]:.3e} vs one ulp {u:.3f}")
        print(f"           -> {'a NEAR-TIE: expected, not a bug' if gap[k] < 4 * u else 'NOT a near-tie: investigate'}")
    print()

print("=" * 72)
print("B6 VERDICT (teacher-forced):", "PASS" if ok else "*** FAIL ***")
print("""
TEST 1 is the verdict.  It drives both paths along identical tokens, so any
difference is pure arithmetic and the answer should be within a bf16 ulp.

TEST 2 is informational.  Greedy decoding amplifies a sub-ulp difference into
a completely different continuation the moment two candidates are tied, so a
divergence there with a tiny top-2 gap is the expected behaviour of a chaotic
map, not evidence of a broken cache.
""")