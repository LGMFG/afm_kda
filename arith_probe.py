#!/usr/bin/env python
"""Arithmetic probe v2 -- copy control + digit-length sweep, one paired design.

WHY THIS EXISTS
---------------
The first probe ("What is a OP b? Answer:", 200 problems) gave teacher 76.0 %,
R1 20.0 %, R2 7.5 %, R3 24.0 %, R4 9.5 %, and 0.20^2.28 reproduced R1's GSM8K.
It could NOT tell apart two very different diagnoses:

  (A) the student cannot COMPUTE   -- the digits arrive, the arithmetic fails
  (B) the student cannot READ      -- it fails to pull the operands out of the
                                      prompt exactly, so there is nothing to compute

Those point at different stage-2 designs.  This script measures both in one job.

CONDITIONS  (200 problems each, identical for every model -> PAIRED comparisons)
----------
  copy_d    "What is 437? Answer: 437"         d = 1..4   reading + writing, NO computation
  add_d     "What is 437 + 562? Answer: 999"   d = 1..4   half with a carry, half without
  sub_d     "What is 562 - 437? Answer: 125"   d = 1..4   half with a borrow, half without (d>=2)
  mul_1     1-digit x 1-digit                              a memorised table
  mul_2x1   2-digit x 1-digit                              one carry chain
  mul_2     2-digit x 2-digit                              partial products

  copy_d minus add_d at the same d isolates COMPUTATION from READING.
  add_1 .. add_4 is the digit-length sweep: does the student fall faster than
  the teacher as more positions must be aligned exactly?
  carry vs no-carry at the same d isolates carry PROPAGATION from digit lookup.

Every condition gets its own 4 exemplars (same op, same d), drawn from a
separate RNG and excluded from the test set.  Greedy, max_new_tokens=12,
use_cache=True, scored on the FIRST integer in the continuation.

⚠️ TOKENISATION DECIDES WHAT "DIGIT ALIGNMENT" MEANS.  If AFM's tokenizer
packs up to three digits into one token, then for d <= 3 there is no
per-digit alignment inside the prompt at all, and the positional explanation
offered so far is wrong for those d.  The script prints how numbers of each
length tokenize BEFORE anything else.  Read that block first.

⚠️ This is a NEW probe.  Its numbers are not the old probe's numbers, because
the old op mix and operand ranges are not recorded.  Every model, including R1
and the teacher, is re-measured here, so all comparisons are internal.

USAGE
-----
  # free, login node, no GPU: build every prompt, print examples, exit
  python arith_probe.py --dry-run

  # one model, 10 problems per condition -- a pipeline check
  python arith_probe.py --smoke 10 --out $DIR --model r1=/path/to/ckpt

  # the real thing (see arith_probe.sbatch)
  python arith_probe.py --out $DIR --model teacher=/path --model r1=/path ...

  Results:  $DIR/<name>.jsonl         one row per problem
            $DIR/<name>.summary.json  per-condition accuracy, timing, tokenisation
  A model whose summary.json exists is SKIPPED, so a job killed by the
  30-minute debug limit is finished by resubmitting the same command.
"""
import argparse
import json
import os
import random
import re
import sys
import time
import zlib

SEED = 20261006
N_PER_COND = 200
N_SHOT = 4
MAX_NEW = 12

CONDITIONS = (
    [("copy", d) for d in (1, 2, 3, 4)]
    + [("add", d) for d in (1, 2, 3, 4)]
    + [("sub", d) for d in (1, 2, 3, 4)]
    + [("mul", "1"), ("mul", "2x1"), ("mul", "2")]
)
SYMBOL = {"add": "+", "sub": "-", "mul": "*"}


# --------------------------------------------------------------------------
# problem generation -- pure python, testable without torch
# --------------------------------------------------------------------------
def cond_name(op, d):
    return f"{op}_{d}"


def rand_d(rng, d):
    """A uniformly random d-digit number (1..9 for d = 1)."""
    lo = 1 if d == 1 else 10 ** (d - 1)
    return rng.randint(lo, 10 ** d - 1)


def has_carry(a, b):
    while a or b:
        if a % 10 + b % 10 >= 10:
            return True
        a //= 10
        b //= 10
    return False


def has_borrow(a, b):
    """a >= b assumed.  True iff some column needs to borrow."""
    while b:
        if a % 10 < b % 10:
            return True
        a //= 10
        b //= 10
    return False


def gen_one(rng, op, d, want_cls):
    """Draw one problem of the requested class.  Returns dict or None."""
    if op == "copy":
        a = rand_d(rng, d)
        return dict(a=a, b=None, target=a, cls="-")
    if op == "add":
        for _ in range(10_000):
            a, b = rand_d(rng, d), rand_d(rng, d)
            cls = "carry" if has_carry(a, b) else "nocarry"
            if want_cls in (None, cls):
                return dict(a=a, b=b, target=a + b, cls=cls)
        return None
    if op == "sub":
        for _ in range(10_000):
            a, b = rand_d(rng, d), rand_d(rng, d)
            if a < b:
                a, b = b, a
            cls = "borrow" if has_borrow(a, b) else "noborrow"
            if want_cls in (None, cls):
                return dict(a=a, b=b, target=a - b, cls=cls)
        return None
    if op == "mul":
        da, db = {"1": (1, 1), "2x1": (2, 1), "2": (2, 2)}[d]
        a, b = rand_d(rng, da), rand_d(rng, db)
        return dict(a=a, b=b, target=a * b, cls="-")
    raise ValueError(op)


def classes_for(op, d):
    """Which classes to balance 50/50.  [None] means: no balancing."""
    if op == "add":
        return ["carry", "nocarry"]
    if op == "sub" and d != 1:          # 1-digit a>=b never borrows
        return ["borrow", "noborrow"]
    return [None]


def enumerate_small(op, d):
    """The complete problem space when it is too small to sample 200 from."""
    def digits(k):
        return range(1, 10) if k == 1 else range(10 ** (k - 1), 10 ** k)
    if op == "copy" and d in (1, 2):
        return [dict(a=a, b=None, target=a, cls="-") for a in digits(d)]
    if op == "add" and d == 1:
        return [dict(a=a, b=b, target=a + b, cls="carry" if a + b >= 10 else "nocarry")
                for a in digits(1) for b in digits(1)]
    if op == "sub" and d == 1:
        return [dict(a=a, b=b, target=a - b, cls="noborrow")
                for a in digits(1) for b in digits(1) if a >= b]
    if op == "mul" and d == "1":
        return [dict(a=a, b=b, target=a * b, cls="-") for a in digits(1) for b in digits(1)]
    return None


def question(op, a, b):
    if op == "copy":
        return f"What is {a}? Answer:"
    return f"What is {a} {SYMBOL[op]} {b}? Answer:"


def make_condition(op, d, n=N_PER_COND, n_shot=N_SHOT, seed=SEED):
    """Exemplars and test problems for one condition, fully deterministic."""
    name = cond_name(op, d)
    salt = zlib.crc32(name.encode())
    ex_rng = random.Random(seed * 7919 + salt)        # exemplars: own stream
    te_rng = random.Random(seed * 104729 + salt)      # test set:  own stream
    classes = classes_for(op, d)

    exemplars, seen = [], set()
    i = 0
    while len(exemplars) < n_shot:
        p = gen_one(ex_rng, op, d, classes[i % len(classes)])
        i += 1
        key = (p["a"], p["b"])
        if key in seen:
            continue
        seen.add(key)
        exemplars.append(p)

    small = enumerate_small(op, d)
    if small is not None:
        # ⭐ Tiny problem spaces (copy_1 has 9 members, add_1 has 81) cannot
        # supply 200 DISTINCT problems, and a repeated prompt under greedy
        # decoding is a repeated answer, not new information.  Test the whole
        # space instead -- every member that is not an exemplar.  n differs
        # per condition as a result; the summary records it.
        problems = [p for p in small if (p["a"], p["b"]) not in seen]
        te_rng.shuffle(problems)
        problems = problems[:n]
    else:
        problems, guard = [], 0
        while len(problems) < n:
            guard += 1
            if guard > 1_000_000:
                raise RuntimeError(f"{name}: cannot draw {n} distinct problems")
            p = gen_one(te_rng, op, d, classes[len(problems) % len(classes)])
            key = (p["a"], p["b"])
            if key in seen:             # never test on an exemplar, never repeat
                continue
            seen.add(key)
            problems.append(p)

    shots = "".join(f"{question(op, e['a'], e['b'])} {e['target']}\n" for e in exemplars)
    for k, p in enumerate(problems):
        p.update(cond=name, op=op, d=str(d), idx=k,
                 prompt=shots + question(op, p["a"], p["b"]))
    return exemplars, problems


def all_problems(n=N_PER_COND):
    out = []
    for op, d in CONDITIONS:
        _, probs = make_condition(op, d, n=n)
        out.extend(probs)
    return out


_INT = re.compile(r"-?\d+")


def parse_first_int(text):
    """First integer in the continuation; thousands separators tolerated."""
    m = _INT.search(text.replace(",", ""))
    return int(m.group(0)) if m else None


# --------------------------------------------------------------------------
# model side -- imports torch only when actually run
# --------------------------------------------------------------------------
def tokenisation_report(tok, rng_seed=7):
    rng = random.Random(rng_seed)
    lines, info = [], {}
    for d in (1, 2, 3, 4, 5):
        nums = [rand_d(rng, d) for _ in range(200)]
        # the number as it appears in the prompt: after "is " / "Answer: "
        counts = [len(tok.tokenize(f" {x}")) for x in nums]
        ex = nums[0]
        info[d] = dict(mean_tokens=sum(counts) / len(counts),
                       min=min(counts), max=max(counts),
                       example=str(ex), pieces=tok.tokenize(f" {ex}"))
        lines.append(f"    {d}-digit: {info[d]['mean_tokens']:.2f} tokens "
                     f"(min {min(counts)}, max {max(counts)})   e.g. ' {ex}' -> {info[d]['pieces']}")
    return lines, info


def run_model(name, path, problems, out_dir, batch_size, check_batching,
              tok_path, suffix=""):
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    # ⭐ ONE tokenizer for every model.  The students were trained with the
    # teacher's tokenizer ("Using teacher's tokenizer" in every training log),
    # so this is correct by construction -- and it guarantees every model sees
    # byte-identical token ids, which is what makes the comparison paired.
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if os.path.exists(os.path.join(path, "tokenizer.json")):
        own = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        probe_txt = "What is 4372 + 56? Answer: 4428\n"
        if own(probe_txt).input_ids != tok(probe_txt).input_ids:
            print(f"    ⚠️⚠️ {name}'s own tokenizer encodes differently from --tokenizer. "
                  "Using --tokenizer; find out why.", flush=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True).to("cuda").eval()
    t_load = time.time() - t0
    print(f"    loaded {type(model).__name__} in {t_load:.0f}s", flush=True)

    tok_lines, tok_info = tokenisation_report(tok)
    print("    tokenisation of numbers (read this before interpreting digit effects):")
    print("\n".join(tok_lines), flush=True)

    # ⭐ No padding anywhere: prompts are bucketed by exact token length, so a
    # batch of B rows runs the same code path as B separate batch-1 calls.
    enc = [tok(p["prompt"]).input_ids for p in problems]
    buckets = {}
    for i, ids in enumerate(enc):
        buckets.setdefault(len(ids), []).append(i)

    @torch.no_grad()
    def generate(rows):
        ids = torch.tensor([enc[i] for i in rows], device="cuda")
        out = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=MAX_NEW, do_sample=False, use_cache=True,
            pad_token_id=pad_id)
        return [tok.decode(o[ids.shape[1]:], skip_special_tokens=True) for o in out]

    gens = [None] * len(problems)
    t1 = time.time()
    n_batches = 0
    for L in sorted(buckets):
        rows = buckets[L]
        for s in range(0, len(rows), batch_size):
            chunk = rows[s:s + batch_size]
            for i, g in zip(chunk, generate(chunk)):
                gens[i] = g
            n_batches += 1
    t_gen = time.time() - t1
    print(f"    generated {len(problems)} in {t_gen:.0f}s ({n_batches} batches, "
          f"{len(buckets)} length buckets)", flush=True)

    # ⭐ The slow oracle: the first 8 prompts of the largest bucket, one at a
    # time.  Must match the batched text exactly; a mismatch means batching is
    # changing answers and every number below is suspect.
    mism = 0
    if check_batching:
        big = max(buckets.values(), key=len)[:8]
        for i in big:
            if generate([i])[0] != gens[i]:
                mism += 1
        print(f"    batch-vs-single check: {len(big) - mism}/{len(big)} identical"
              + ("" if mism == 0 else "   ⚠️⚠️ BATCHING CHANGES OUTPUTS"), flush=True)

    rows_out, per = [], {}
    for p, g in zip(problems, gens):
        pred = parse_first_int(g)
        ok = pred == p["target"]
        rows_out.append(dict(model=name, cond=p["cond"], op=p["op"], d=p["d"],
                             cls=p["cls"], idx=p["idx"], a=p["a"], b=p["b"],
                             target=p["target"], pred=pred, correct=ok,
                             gen=g[:40]))
        c = per.setdefault(p["cond"], [0, 0])
        c[0] += ok
        c[1] += 1

    tag = name + suffix
    with open(os.path.join(out_dir, f"{tag}.jsonl"), "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    summary = dict(model=name, path=path, seed=SEED,
                   accuracy={k: v[0] / v[1] for k, v in per.items()},
                   n={k: v[1] for k, v in per.items()},
                   batch_vs_single_mismatches=mism,
                   tokenisation={str(k): v for k, v in tok_info.items()},
                   seconds=dict(load=round(t_load), generate=round(t_gen)))
    with open(os.path.join(out_dir, f"{tag}.summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print("    " + "  ".join(f"{k}:{100 * v[0] / v[1]:.0f}" for k, v in per.items()), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer for ALL models -- the teacher's (default: the "
                         "model named 'teacher', else the first --model)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--smoke", type=int, default=0, help="N problems per condition")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-check-batching", action="store_true")
    args = ap.parse_args()

    n = args.smoke or N_PER_COND
    problems = all_problems(n)
    print(f"{len(problems)} problems per model across {len(CONDITIONS)} conditions "
          f"(up to {n} each; tiny spaces are tested exhaustively), "
          f"{N_SHOT}-shot, greedy, max_new_tokens={MAX_NEW}, seed {SEED}")

    if args.dry_run:
        for op, d in CONDITIONS:
            ex, probs = make_condition(op, d, n=n)
            cls = {}
            for p in probs:
                cls[p["cls"]] = cls.get(p["cls"], 0) + 1
            print(f"\n=== {cond_name(op, d)}   classes {cls}")
            print("    " + probs[0]["prompt"].replace("\n", "\n    ")
                  + f"   -> expects {probs[0]['target']}")
        print("\n✅ dry run: prompts build, nothing loaded.")
        return

    if not args.out or not args.model:
        sys.exit("need --out DIR and at least one --model NAME=PATH")
    os.makedirs(args.out, exist_ok=True)

    # ⭐ Fail before loading anything if any path is wrong.
    models, bad = [], []
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not name or not path:
            sys.exit(f"bad --model {spec!r}; use NAME=PATH")
        if not os.path.isfile(os.path.join(path, "config.json")):
            bad.append(f"{name}: {path}")
        models.append((name, path))
    if bad:
        sys.exit("FATAL: no config.json at:\n  " + "\n  ".join(bad)
                 + "\nFix the paths (ls $WORK_LUKAS/output/) and resubmit.")

    tok_path = args.tokenizer or dict(models).get("teacher") or models[0][1]
    print(f"tokenizer for all models: {tok_path}")

    for name, path in models:
        if args.smoke == 0 and os.path.exists(os.path.join(args.out, f"{name}.summary.json")):
            print(f"\n=== {name}: summary exists, SKIPPED")
            continue
        print(f"\n=== {name}   {path}", flush=True)
        # smoke output gets its own file names, so it can never make the
        # full run SKIP a model
        run_model(name, path, problems, args.out, args.batch_size,
                  check_batching=not args.no_check_batching,
                  tok_path=tok_path,
                  suffix=".smoke" if args.smoke else "")
    print("\nDONE")


if __name__ == "__main__":
    main()