#!/usr/bin/env python
"""Verify a trained checkpoint is ready to GENERATE, and flip `use_cache`.

PROVENANCE MODEL
----------------
The corrected modeling_afm_kda.py lives in afm_kda/ under git.  The
students/*_init directories are built from it.  Every training run loads an
init student, and save_pretrained carries the custom .py files from the loaded
directory into the output.  So each checkpoint ALREADY holds the code it was
trained with, and the afm_kda commit identifies which version that is.
Nothing needs copying.  That is correctness by construction rather than by
remembering, which is why this script does NOT copy by default.

WHAT IS STILL NEEDED AFTER EVERY RUN
------------------------------------
`use_cache`.  HF Trainer sets config.use_cache = False whenever gradient
checkpointing is on, and save_pretrained writes that out -- regardless of what
the init student's config said, so a correct init does not buy you a correct
output config.  Left false, generation recomputes the whole prefix at every
token: still correct, but O(n^2) instead of O(n), and it makes any
decode-throughput measurement (B9b) measure the wrong thing entirely.
It does NOT affect likelihood-scored benchmarks (ARC-C, HellaSwag,
Winogrande, MMLU), which never generate.

WHAT IS STILL WORTH CHECKING
----------------------------
The provenance chain above is silent when it breaks: an init student that was
not rebuilt after a fix, a run launched from an older init directory, an edit
that never landed.  The MARKERS below turn that into an error message instead
of a wrong number three weeks later.

USAGE
-----
    python finalize_checkpoint.py $WORK_LUKAS/output/<run> [<run>/checkpoint-NNNN ...]
    python finalize_checkpoint.py --sync <dir> ...     # retrofit a PRE-FIX checkpoint
    python finalize_checkpoint.py --dry-run <dir> ...  # report only, change nothing

--sync copies afm_kda/*.py over the checkpoint's own copies.  It exists for
checkpoints trained before a fix existed.  Under the provenance model above a
fresh run should never need it, and needing it is itself a finding -- so it is
opt-in, and the script says so loudly when it is used.

Exits non-zero if any directory fails a check, so it can be the last line of a
training sbatch.
"""
import json
import os
import shutil
import subprocess
import sys

SRC = os.environ.get("AFM_KDA_SRC", os.path.join(os.environ.get("WORK_LUKAS", ""), "afm_kda"))
CODE = ("modeling_afm_kda.py", "configuration_afm_kda.py")

# One line per fix that must be present in the checkpoint's modelling file.
# Add to this every time a fix lands; a stale checkpoint then announces itself.
MARKERS = {
    "class AFMKDACache":  "B2' -- the two-cache bridge exists",
    "def get_seq_length": "B2' -- how many tokens are behind us",
    "def get_mask_sizes": "B7  -- how wide the causal mask must be",
    "_padding_mask_2d":   "B7  -- the 2-D padding mask reaches the KDA layers",
    "_is_stateful":       "declares the model unsuitable for assisted generation",
}


def git_provenance(d):
    """Report the afm_kda commit this checkpoint records, if the sbatch wrote one."""
    stamp = os.path.join(d, "afm_kda_commit.txt")
    dirty = os.path.join(d, "afm_kda_dirty.txt")
    if not os.path.exists(stamp):
        print("    (no afm_kda_commit.txt -- add the git stamp to the training sbatch)")
        return
    sha = open(stamp).read().strip()[:12]
    if os.path.exists(dirty) and open(dirty).read().strip():
        print(f"    !! commit {sha} recorded, but the tree was DIRTY when it ran "
              f"-- the hash does not describe the code")
    else:
        print(f"    afm_kda commit {sha}, clean tree")


def finalise(d, sync=False, dry=False):
    ok = True
    print(f"\n=== {d}")
    if not os.path.isdir(d):
        print("    !! MISSING directory")
        return False

    if sync:
        print("    !! --sync: overwriting the checkpoint's own code copies.")
        print("       A fresh run should not need this.  Note why it did.")
        for f in CODE:
            s = os.path.join(SRC, f)
            if os.path.exists(s):
                if not dry:
                    shutil.copy2(s, os.path.join(d, f))
                print(f"       copied {f}")
            else:
                print(f"    !! source missing: {s}")
                ok = False

    # 1. is the code in this checkpoint the code we think it is?
    p = os.path.join(d, "modeling_afm_kda.py")
    if not os.path.exists(p):
        print("    !! modeling_afm_kda.py ABSENT -- this checkpoint cannot load")
        return False
    text = open(p).read()
    missing = [(m, why) for m, why in MARKERS.items() if m not in text]
    for m, why in missing:
        print(f"    !! MARKER ABSENT: {m:22} ({why})")
    if missing:
        print("       -> trained before that fix, or from a stale init student.")
        print("          Re-run with --sync only if you accept that the weights")
        print("          were produced by different code than now ships with them.")
        ok = False
    else:
        print(f"    all {len(MARKERS)} markers present")

    git_provenance(d)

    # 2. auto_map must point at our files, or transformers loads Arcee's classes
    cfg = os.path.join(d, "config.json")
    if os.path.exists(cfg):
        am = json.load(open(cfg)).get("auto_map", {})
        if not any("modeling_afm_kda" in str(v) for v in am.values()):
            print(f"    !! auto_map does not reference modeling_afm_kda: {am}")
            ok = False
        else:
            print("    auto_map ok")

    # 3. the flip
    for f in ("config.json", "generation_config.json"):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            print(f"    (no {f})")
            continue
        c = json.load(open(p))
        before = c.get("use_cache")
        if before is True:
            print(f"    {f}: use_cache already True")
            continue
        c["use_cache"] = True
        if not dry:
            json.dump(c, open(p, "w"), indent=2)
        print(f"    {f}: use_cache {before!r} -> True{' (dry-run)' if dry else ''}")

    return ok


if __name__ == "__main__":
    args = sys.argv[1:]
    sync = "--sync" in args
    dry = "--dry-run" in args
    dirs = [a for a in args if not a.startswith("--")]
    if not dirs:
        print(__doc__)
        sys.exit(2)
    if not SRC and sync:
        print("!! WORK_LUKAS unset and AFM_KDA_SRC unset -- cannot --sync")
        sys.exit(2)

    results = {d: finalise(d, sync=sync, dry=dry) for d in dirs}
    print("\n" + "=" * 64)
    for d, r in results.items():
        print(f"  {'PASS' if r else 'FAIL'}  {d}")
    print("\nIf anything was copied, clear $HF_HOME/modules/transformers_modules")
    print("before the next load, or the OLD code is what runs.")
    sys.exit(0 if all(results.values()) else 1)