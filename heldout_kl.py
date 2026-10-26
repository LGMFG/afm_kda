#!/usr/bin/env python
"""Held-out CE and KL against the teacher, on BOTH corpora, for every model -- one table.

    python heldout_kl.py                      # default model list (below), skips models already done
    MODELS_FILE=list.txt python heldout_kl.py # any list, one name=path per line (e.g. checkpoint curves)
    python heldout_kl.py --table              # only print the table from the saved results

Why: every training run so far reported eval_loss on ONE held-out set (FineWeb runs on FineWeb-Edu,
FineMath runs on FineMath), so the two families cannot be compared from their logs, and no run
measured the other corpus.  This script puts all models on the same two sets:

    web   fineweb_edu_3B_heldout    -> how well a model still matches the teacher on general web text
                                       (for the FineMath runs: FORGETTING, against R1 / R1-S2)
    math  finemath_4plus_heldout    -> how well it matches the teacher on maths text
                                       (for the FineWeb runs: the off-distribution gap; for FineMath: the gain)

Per model and corpus: CE (vs the true next token), KL(teacher || student) at T = 2 with DistillKit's
own definition (kl.py: sum over vocab of KL on T-softened distributions, x T^2, / tokens), and the
per-sequence CE gap to the teacher (median, worst, number of sequences more than +0.5 nat above the
teacher) -- the ce_outliers check for latent damage, built in.

Sequences: 64 x 2048 tokens per corpus, documents concatenated in file order with the tokenizer's
special tokens -- exactly test_r1_ce.py's construction, so the web row must reproduce
teacher CE 2.2963 and R1 CE 2.4559 / KL 0.3568 (test_r5_kl.py).  The script checks this.

Results: one JSON per model in $WORK_LUKAS/eval_results/heldout_kl/; a model whose JSON exists is
skipped, so a job killed at the debug 30-minute limit is finished by resubmitting the same line.

LAYOUT=docs (diagnostic, v2): each sequence is ONE document from its start, as DistillKit trains and
evaluates -- [BOS] + text + [EOS], cut at 2048, no neighbours, no mid-document start (the first 64
documents of each held-out file).  Results go to eval_results/heldout_kl_docs/; the default
(concatenated) layout and its directory are unchanged.  Use: does a model's tail of bad sequences
survive when the input looks exactly like training?  (No validity reference exists for this layout.)

Two more diagnostic layouts (v3) isolate WHICH difference from training matters -- both are the
default concatenated stream, changed in one respect only:
  LAYOUT=concat_bos  every window starts with BOS: [BOS] + 2047 tokens of the stream
                     (fixes the mid-document window start; junctions stay "text<BOS>text", no EOS)
  LAYOUT=concat_eos  EOS appended to every document before concatenating: junctions become
                     "text<EOS><BOS>text" as in DistillKit's packing; windows still start mid-document
Results go to eval_results/heldout_kl_<layout>/.
"""
import argparse
import gc
import json
import math
import os
import statistics as st

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

W = os.environ.get("WORK_LUKAS", "/leonardo_work/EUHPC_D34_049/WORK_LUKAS")
O = f"{W}/output"
TEACHER = f"{W}/hf_cache/hub/models--arcee-ai--AFM-4.5B-Base/snapshots/dab7922f9c868d479b365410304f466b007b1c5a"
CORPORA = {"web": f"{W}/datasets/fineweb_edu_3B_heldout", "math": f"{W}/datasets/finemath_4plus_heldout"}
LAYOUT = os.environ.get("LAYOUT", "concat")
assert LAYOUT in ("concat", "docs", "concat_bos", "concat_eos"), LAYOUT
OUTDIR = f"{W}/eval_results/heldout_kl" + ("" if LAYOUT == "concat" else f"_{LAYOUT}")
SEQ, N_SEQ, T = 2048, int(os.environ.get("N_SEQ", 64)), 2.0
DEV, DTYPE, IMPL = "cuda", torch.bfloat16, "sdpa"
VALID = {"teacher": {"web_ce": 2.2963}, "r1_kda_hybrid": {"web_ce": 2.4559, "web_kl": 0.3568}}

DEFAULT = [
    ("teacher", TEACHER),
    ("r1_kda_hybrid", f"{O}/r1_stage1_hybrid31/checkpoint-4580"),
    ("r2_gdn_hybrid", f"{O}/r2_stage1_gdn_hybrid31/checkpoint-4580"),
    ("r3_kda_only", f"{O}/r3_stage1_kda_only/checkpoint-4580"),
    ("r4_gdn_only", f"{O}/r4_stage1_gdn_only/checkpoint-4580"),
    ("r1_s2_kda_hyb", f"{O}/r1_stage2_kda_hybrid31_unfreeze/checkpoint-15260"),
    ("r2_s2_gdn_hyb", f"{O}/r2_stage2_gdn_hybrid31_unfreeze/checkpoint-15260"),
    ("r3_s2_kda_only", f"{O}/r3_stage2_kda_only_unfreeze/checkpoint-15260"),
    ("r4_s2_gdn_only", f"{O}/r4_stage2_gdn_only_unfreeze/checkpoint-15260"),
    ("r1_s2cos_kda_hyb", f"{O}/r1_stage2_kda_hybrid31_cosine5e5/checkpoint-15260"),
    ("r1_s1math_kda_hyb", f"{O}/r1_stage1_hybrid31_finemath/checkpoint-4580"),
    ("r1_s2math_kda_hyb", f"{O}/r1_stage2_kda_hybrid31_finemath/checkpoint-15260"),
    ("r1fm_s2math_kda_hyb", f"{O}/r1fm_stage2_kda_hybrid31_finemath/checkpoint-15260"),
    ("r2_s2math_gdn_hyb", f"{O}/r2_stage2_gdn_hybrid31_finemath/checkpoint-15260"),
    ("r3_s2math_kda_only", f"{O}/r3_stage2_kda_only_finemath/checkpoint-15260"),
    ("r4_s2math_gdn_only", f"{O}/r4_stage2_gdn_only_finemath/checkpoint-15260"),
]


def build(tok, path):
    """concat: one tensor [N_SEQ, SEQ].  docs: a list of N_SEQ 1-D tensors (one document each)."""
    ds = load_from_disk(path)
    if LAYOUT == "docs":
        seqs = []
        for row in ds:
            ids = (tok(row["text"]).input_ids + [tok.eos_token_id])[:SEQ]
            if len(ids) >= 64:                    # skip near-empty documents
                seqs.append(torch.tensor(ids))
            if len(seqs) == N_SEQ:
                return seqs
        raise AssertionError(f"{path}: only {len(seqs)} documents")
    L = SEQ - 1 if LAYOUT == "concat_bos" else SEQ       # window length before any BOS is added
    buf = []
    for row in ds:
        buf.extend(tok(row["text"]).input_ids + ([tok.eos_token_id] if LAYOUT == "concat_eos" else []))
        if len(buf) >= L * N_SEQ:
            break
    assert len(buf) >= L * N_SEQ, f"{path}: only {len(buf)} tokens"
    b = torch.tensor(buf[: L * N_SEQ]).view(N_SEQ, L)
    if LAYOUT == "concat_bos":
        b = torch.cat([torch.full((N_SEQ, 1), tok.bos_token_id, dtype=b.dtype), b], dim=1)
    return b


def load(path):
    return AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE, attn_implementation=IMPL,
                                                trust_remote_code=True).to(DEV).eval()


@torch.no_grad()
def measure(student, teacher, batch):
    """per-sequence CE (student, teacher) and token-mean KL(T) for one corpus."""
    ce_s, ce_t, kl, bad = [], [], [], 0
    for i in range(len(batch)):
        ids = batch[i].view(1, -1).to(DEV)
        tgt = ids[0, 1:]
        ls = student(input_ids=ids, use_cache=False).logits[0, :-1].float()
        bad += int((~torch.isfinite(ls)).any(-1).sum())
        ce_s.append(torch.nn.functional.cross_entropy(ls, tgt).item())
        if teacher is None:                       # the teacher itself: KL = 0 by definition
            ce_t.append(ce_s[-1]); kl.append(0.0)
            continue
        lt = teacher(input_ids=ids, use_cache=False).logits[0, :-1].float()
        ce_t.append(torch.nn.functional.cross_entropy(lt, tgt).item())
        k = 0.0
        for a in range(0, ls.shape[0], 512):      # chunked: full-vocab log-softmax is ~1 GB per 2k tokens
            s = torch.log_softmax(ls[a:a + 512] / T, -1)
            t = torch.log_softmax(lt[a:a + 512] / T, -1)
            k += torch.nn.functional.kl_div(s, t, reduction="sum", log_target=True).item()
        kl.append(k * T * T / ls.shape[0])
        del ls, lt
    gap = [a - b for a, b in zip(ce_s, ce_t)]
    return {"ce": st.mean(ce_s), "ce_teacher": st.mean(ce_t), "kl": st.mean(kl),
            "gap": st.mean(gap), "gap_median": st.median(gap), "gap_worst": max(gap),
            "gap_paired_se": st.stdev(gap) / math.sqrt(len(gap)) if len(gap) > 1 else float("nan"),
            "n_seq_gap_gt_0p5": sum(g > 0.5 for g in gap), "nonfinite_rows": bad,
            "per_seq_ce": ce_s, "per_seq_kl": kl}


def table():
    rows = []
    for f in sorted(os.listdir(OUTDIR)) if os.path.isdir(OUTDIR) else []:
        if f.endswith(".json"):
            rows.append(json.load(open(os.path.join(OUTDIR, f))))
    order = [n for n, _ in DEFAULT]
    rows.sort(key=lambda r: (order.index(r["name"]) if r["name"] in order else 99, r["name"]))
    print(f"\n{'model':26}{'web CE':>9}{'web KL':>9}{'math CE':>9}{'math KL':>9}{'math/web KL':>12}"
          f"{'worst gap w/m':>15}{'#>0.5 w/m':>11}")
    for r in rows:
        w, m = r["web"], r["math"]
        ratio = m["kl"] / w["kl"] if w["kl"] > 0 else float("nan")
        print(f"{r['name'][:25]:26}{w['ce']:9.4f}{w['kl']:9.4f}{m['ce']:9.4f}{m['kl']:9.4f}{ratio:12.2f}"
              f"{w['gap_worst']:8.2f}/{m['gap_worst']:<6.2f}{w['n_seq_gap_gt_0p5']:6}/{m['n_seq_gap_gt_0p5']:<4}")
    print("\nKL = KL(teacher || student) at T = 2, x T^2, token mean (DistillKit's definition).  CE = vs true token.")
    print("worst gap = largest per-sequence CE excess over the teacher (nat); #>0.5 = sequences > +0.5 nat (R5 had 4/64 web).")
    for name, ref in (VALID.items() if LAYOUT == "concat" else []):
        r = next((x for x in rows if x["name"] == name), None)
        if r:
            ok = all(abs(r["web"][k.split("_")[1]] - v) < 2e-3 for k, v in ref.items())
            print(f"validity {name}: " + ", ".join(f"{k} {r['web'][k.split('_')[1]]:.4f} (ref {v})" for k, v in ref.items())
                  + ("   OK" if ok else "   !! MISMATCH -- not on the established axis"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if a.table:
        return table()
    models = DEFAULT
    if os.environ.get("MODELS_FILE"):
        models = [tuple(l.strip().split("=", 1)) for l in open(os.environ["MODELS_FILE"]) if "=" in l]
    todo = [(n, p) for n, p in models
            if not os.path.exists(f"{OUTDIR}/{n}.json") and (os.path.isdir(p) or print(f"skip {n}: no {p}"))]
    print(f"layout {LAYOUT} -> {OUTDIR}")
    print(f"{len(todo)} models to measure, {N_SEQ} x {SEQ} tokens per corpus: {[n for n, _ in todo]}", flush=True)
    if todo:
        tok = AutoTokenizer.from_pretrained(TEACHER)
        batches = {c: build(tok, p) for c, p in CORPORA.items()}
        teacher = load(TEACHER)
        for name, path in todo:
            is_t = os.path.realpath(path) == os.path.realpath(TEACHER)
            m = teacher if is_t else load(path)
            res = {"name": name, "path": path}
            for c, b in batches.items():
                res[c] = measure(m, None if is_t else teacher, b)
                r = res[c]
                print(f"  {name:24} {c:4}  CE {r['ce']:.4f}  KL {r['kl']:.4f}  gap {r['gap']:+.4f} "
                      f"(median {r['gap_median']:+.4f}, worst {r['gap_worst']:+.3f}, >0.5: {r['n_seq_gap_gt_0p5']})", flush=True)
            json.dump(res, open(f"{OUTDIR}/{name}.json", "w"))
            if not is_t:
                del m
            gc.collect(); torch.cuda.empty_cache()
    table()


if __name__ == "__main__":
    main()