#!/usr/bin/env python
"""Is the damage LOCALISED to mathematical reasoning, or is it uniform?

GSM8K reads ~0 while ARC-C, HellaSwag and WinoGrande retain 74-89 % of the
teacher's above-chance score.  Two readings fit:

  (1) uniform degradation.  The student is a slightly worse language model
      everywhere (+0.1595 nats on FineWeb-Edu), and GSM8K collapses only
      because exact-match over a multi-step chain amplifies a small penalty.
  (2) localised damage.  Mathematical text specifically is harder for the
      student than its overall perplexity penalty would predict.

These have different consequences for the next training run, and they are
distinguishable by one cheap measurement: compare the student-vs-teacher CE
gap on MATH text against the same gap on WEB text.

  gap_math ~= gap_web   -> reading (1).  Nothing special about math; the
                           benchmark collapse is the compounding argument.
  gap_math >> gap_web   -> reading (2).  Mathematical structure is what the
                           conversion cost, which is a sharper claim and a
                           stronger argument for the unfrozen-MLP stage.

⭐ BOTH corpora are measured in the same process, same dtype, same kernels,
so the comparison is internally controlled.  The web-text row doubles as a
VALIDITY CHECK: it must reproduce test_r1_ce.py's 2.2963 / 2.4559.  If it
does not, the script is not on the established axis and nothing else in the
output can be trusted.

⚠️ The right error bar is the PAIRED per-sequence difference, not the spread
across sequences.  Both models see identical passages, and passage difficulty
dominates the unpaired spread -- test_r1_ce.py's printed +-0.040 is the wrong
statistic for this comparison and is corrected here.

Run:  sbatch test_r1_ce_math.sbatch      (1 GPU, debug qos, ~20 min)
"""
import math
import os

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = os.environ["WORK_LUKAS"]
STUDENT = os.environ.get("MODEL", f"{WORK}/output/r1_stage1_hybrid31")
TEACHER = os.environ.get(
    "TEACHER",
    f"{WORK}/hf_cache/hub/models--arcee-ai--AFM-4.5B-Base/snapshots/"
    "dab7922f9c868d479b365410304f466b007b1c5a",
)
SEQ = 2048
N_SEQ = int(os.environ.get("N_SEQ", 64))          # 64 x 2048 = 131,072 tokens
DEV, DTYPE, IMPL = "cuda", torch.bfloat16, "sdpa"

CORPORA = {
    "web  (FineWeb-Edu held-out)": f"{WORK}/datasets/fineweb_edu_3B_heldout",
    "math (MetaMathQA)":           f"{WORK}/datasets/metamathqa",
}

# ⭐ CONTROL 1 -- a third corpus that is OUT-OF-DISTRIBUTION but NOT mathematical.
# The 2.11x result cannot tell "maths is damaged" apart from "anything unlike
# FineWeb-Edu is damaged", because the student was trained to match the teacher
# on FineWeb-Edu and maths is off that distribution.  A non-maths OOD corpus
# separates them:  gap ~= web  -> maths is special;  gap ~= maths -> it is just
# distance from the training distribution.
#   EXTRA_CORPUS="code (the-stack):/path/to/dataset"
if os.environ.get("EXTRA_CORPUS"):
    _name, _, _path = os.environ["EXTRA_CORPUS"].partition(":")
    CORPORA[_name.strip()] = _path.strip()

# ⭐ CONTROL 2 -- boundary density.  Packing concatenates documents into fixed
# rows without isolation, so a corpus of short documents puts far more document
# boundaries inside each sequence.  MetaMathQA averages 216 tokens/doc against
# FineWeb-Edu's 910, i.e. ~4x the boundaries -- and a fixed recurrent state
# cannot selectively ignore stale context the way attention can.  Raise this to
# keep only longer documents and re-run:  MIN_DOC_TOK=600
MIN_DOC_TOK = int(os.environ.get("MIN_DOC_TOK", 0))

# Column preference, most specific first.  The local MetaMathQA copy turned out
# to be in CHAT form -- one `messages` column holding a list of role/content
# dicts -- so that case is handled explicitly in row_text() below.
COLUMN_SETS = [
    ("messages",),
    ("query", "response"),
    ("question", "answer"),
    ("problem", "solution"),
    ("text",),
]

tok = AutoTokenizer.from_pretrained(TEACHER)


def pick_columns(names):
    """Choose which fields make up one document -- explicitly, never silently."""
    for cand in COLUMN_SETS:
        if all(c in names for c in cand):
            return list(cand)
    raise SystemExit(
        f"Cannot tell which columns hold the text. Available: {names}\n"
        f"Add the right pair to COLUMN_SETS rather than guessing."
    )


def row_text(row, cols):
    """One document as plain text.

    The chat form is flattened to its content only -- no role markers, no
    template.  Both models are base models that never saw a chat format, and
    both see identical tokens either way, so the PAIRED comparison is unaffected;
    dropping the scaffolding just keeps the text closer to mathematical prose.
    """
    if cols == ["messages"]:
        parts = []
        for m in row["messages"]:
            parts.append(str(m.get("content", "")) if isinstance(m, dict) else str(m))
        return "\n".join(p for p in parts if p)
    return "\n".join(str(row[c]) for c in cols if row[c])


def build_batch(path, tag):
    """One [N_SEQ, SEQ] tensor of real tokens from a dataset on disk."""
    ds = load_from_disk(path)
    if hasattr(ds, "keys"):                       # a DatasetDict
        split = "train" if "train" in ds else list(ds.keys())[0]
        print(f"    {tag}: DatasetDict, using split '{split}'")
        ds = ds[split]
    cols = pick_columns(ds.column_names)
    print(f"    {tag}: {len(ds)} rows, columns {ds.column_names} -> using {cols}")

    need = N_SEQ * SEQ + 1
    buf, used, skipped = [], 0, 0
    for row in ds:
        text = row_text(row, cols)
        if MIN_DOC_TOK:
            ids = tok(text).input_ids
            if len(ids) < MIN_DOC_TOK:
                skipped += 1
                continue
        if used == 0:
            # ⭐ Eyeball the first document.  We are inferring the structure of
            # this dataset, and a silent mis-parse would make every number below
            # meaningless while looking perfectly healthy.
            preview = text[:300].replace("\n", " | ")
            print(f"    {tag}: first document looks like:\n        {preview!r}")
        buf.extend(tok(text).input_ids)
        used += 1
        if len(buf) >= need:
            break
    if len(buf) < need:
        raise SystemExit(f"{tag}: only {len(buf)} tokens available, need {need}")
    print(f"    {tag}: {need - 1} tokens from {used} documents "
          f"({(need - 1) / used:.0f} tok/doc, "
          f"{used / N_SEQ:.1f} docs per {SEQ}-token sequence)"
          + (f", {skipped} skipped under MIN_DOC_TOK={MIN_DOC_TOK}" if MIN_DOC_TOK else ""))
    return torch.tensor(buf[: N_SEQ * SEQ], device=DEV).view(N_SEQ, SEQ)


@torch.no_grad()
def per_sequence_ce(model, batch):
    """CE for each sequence separately -- needed for the paired statistic."""
    out = []
    for i in range(batch.shape[0]):
        ids = batch[i: i + 1]
        logits = model(input_ids=ids, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(
            logits[0, :-1].float(), ids[0, 1:], reduction="mean")
        out.append(loss.item())
    return torch.tensor(out)


print(f"{N_SEQ} x {SEQ} = {N_SEQ * SEQ} tokens per corpus\n")
print("Building batches")
batches = {name: build_batch(path, name) for name, path in CORPORA.items()}
print()

ce = {}                                            # ce[model][corpus] -> [N_SEQ]
for tag, path in [("teacher", TEACHER), ("student", STUDENT)]:
    print(f"=== {tag}", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        path, dtype=DTYPE, attn_implementation=IMPL,
        trust_remote_code=True).to(DEV).eval()
    ce[tag] = {}
    for name, batch in batches.items():
        v = per_sequence_ce(m, batch)
        ce[tag][name] = v
        print(f"    {name:28} CE {v.mean():.4f}   ppl {math.exp(v.mean()):8.2f}")
    del m
    torch.cuda.empty_cache()
    print()

print("=" * 78)
print(f"{'corpus':30} {'teacher':>9} {'student':>9} {'gap':>9} {'paired se':>10} {'ppl ratio':>10}")
gaps = {}
for name in batches:
    t, s = ce["teacher"][name], ce["student"][name]
    d = s - t                                      # paired, per sequence
    se = (d.std(unbiased=True) / math.sqrt(len(d))).item()
    gaps[name] = d.mean().item()
    print(f"{name:30} {t.mean():9.4f} {s.mean():9.4f} "
          f"{d.mean():+9.4f} {se:10.4f} {math.exp(d.mean()):10.3f}x")

web = next(k for k in gaps if k.startswith("web"))
mth = next(k for k in gaps if k.startswith("math"))
ratio = gaps[mth] / gaps[web] if gaps[web] else float("nan")

print("\ngap relative to the training distribution (web = 1.00):")
for name in gaps:
    print(f"    {name:30} {gaps[name] / gaps[web]:5.2f}x")

print("\n" + "=" * 78)
print(f"""VALIDITY CHECK
  ⭐ Check the GAP, not the absolute CE.  Any filter (MIN_DOC_TOK) changes
  WHICH documents are sampled, so the absolute level legitimately moves --
  it went 2.2963 -> 2.3927 on web under MIN_DOC_TOK=600.  The student-teacher
  GAP is the quantity that must be stable, and it was: +0.1595 -> +0.1592.

  reference (test_r1_ce.py, unfiltered): teacher 2.2963  student 2.4559  gap +0.1595
  measured here:                         teacher {ce['teacher'][web].mean():.4f}  student {ce['student'][web].mean():.4f}  gap {gaps[web]:+.4f}

  If the GAP disagrees by more than ~0.01, this script is not on the
  established axis and no other row means anything.  Stop and find out why.
  A different absolute CE with the same gap is expected under filtering.

VERDICT
  gap_math / gap_web = {ratio:.2f}x

  ~1.0  -> UNIFORM degradation.  Math text is no harder for the student than
           web text relative to the teacher.  GSM8K's collapse is then purely
           the compounding of a small per-token penalty over a long exact-match
           chain -- and the argument for unfreezing the MLPs rests on the
           reference's ablations rather than on our own measurement.

  >>1.0 -> LOCALISED damage.  Mathematical structure specifically is what the
           conversion cost.  That is a sharper thesis claim, it is measured
           rather than inferred, and it makes the unfrozen-MLP stage the
           directly indicated remedy.

  <1.0  -> would be surprising and worth a second look before believing.

⚠️ This measures LANGUAGE MODELLING on mathematical text, which is not the
same thing as the ABILITY TO SOLVE a problem.  A model can predict the tokens
of a worked solution it could not have produced.  Treat the result as evidence
about where representational damage sits, not as a second GSM8K.

⚠️⚠️ AND A LOCALISED RESULT DOES NOT NAME THE CAUSE.  RESOLVED 2026-09-23:
both controls have now been run, and the second one changed the answer.

  MIN_DOC_TOK=600   boundary density.  The ratio went UP (2.11x -> 2.82x),
      so packed-document boundaries are NOT driving it.  Confound ruled out.

  EXTRA_CORPUS=     a third corpus, out-of-distribution but NOT mathematical.
      Tulu 3 (instruction + code) came back at 2.08x against maths' 2.11x.
      => MATHEMATICS IS NOT SPECIAL.  What the gap tracks is DISTANCE FROM
         THE TRAINING DISTRIBUTION.  Tulu 3's own maths content cannot
         explain this: the gap mixes linearly, so a 1.00x/2.11x blend only
         reaches 2.08x at ~97 % maths, which Tulu 3 is not.

⚠️ And no control here can separate "the architecture generalises badly
off-distribution" from "any student distilled on corpus X generalises badly
off X".  That needs the same recipe on a NON-CONVERTED student.  Report the
ratio as measured; do not attribute it to KDA.

⚠️ Normalisation changes the MAGNITUDE, not the sign.  Raw nats 2.11x,
perplexity ratio 1.401 vs 1.173, gap as a fraction of the teacher's own CE
38.9 % vs 6.9 %.  Quote one, name it, and say the direction is robust to the
choice -- the two corpora have very different entropy (teacher ppl 9.94 vs
2.38) and no single normalisation is obviously the right one.
""")