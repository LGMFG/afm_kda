#!/usr/bin/env python
"""Build an untrained linear-attention student from AFM-4.5B-Base and save it
as a self-contained HuggingFace model directory.

    python afm_kda/build_student.py --variant hybrid31
    python afm_kda/build_student.py --variant hybrid31 --no-transfer   # de-novo ablation
    python afm_kda/build_student.py --variant only --force

Needs ~20 GB of RAM (teacher + student in CPU memory at once) -- run it as a
GPU-node job, NOT on lrd_all_serial, whose 15400 MB cap cannot be raised.
"""
from __future__ import annotations

import argparse, datetime, gc, json, os, shutil, socket, subprocess, sys
import torch

WORK = os.environ["WORK_LUKAS"]
SRC = f"{WORK}/afm_kda"
sys.path.insert(0, WORK)

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from afm_kda.configuration_afm_kda import AFMKDAConfig, build_kda_layers
from afm_kda.modeling_afm_kda import (
    AFMKDAForCausalLM, init_new_parameters, transfer_attention_into_kda,
)

TEACHER = "arcee-ai/AFM-4.5B-Base"
N_LAYERS = 36

# Exact per-layer arithmetic, from the A1 budget.  Used as an assertion, not a guess:
#   4,619,184,640 + 27*11,883,668 + 9*6,553,600 = 4,999,026,076   (measured, A8)
KDA_DELTA = 11_883_668        # KDA layer (27,612,308) - Arcee attention layer (15,728,640)
GATE_PARAMS = 6_553_600       # gate_proj, 2560 x 2560

# auto_map points at these, and HF copies only ONE level of relative imports,
# so the whole import closure has to ship inside the checkpoint directory.
CODE_FILES = ("__init__.py", "configuration_afm_kda.py", "modeling_afm_kda.py",
              "configuration_arcee.py", "modeling_arcee.py")

VARIANTS = {
    # Arcee's KDA-NoPE: nine blocks of "3 KDA + 1 full attention".
    "hybrid31": dict(
        dirname="afm_kda_hybrid31_init",
        full_attn=list(range(3, N_LAYERS, 4)),          # 3,7,11,...,35
        use_rope=False, use_attn_gate=True,
    ),
    # No full attention anywhere.
    "only": dict(
        dirname="afm_kda_only_init",
        full_attn=[],
        use_rope=False, use_attn_gate=True,
    ),
    # "Front-loaded positions": first four layers full attention, nine in total.
    # WARNING: the blog post does not give the remaining five indices -- this
    # spacing is OUR choice.  Check arcee-ai/AFM-4.5B-Base-KDA-FLP/config.json
    # before drawing any comparison with their number.
    "flp": dict(
        dirname="afm_kda_flp_init",
        full_attn=[0, 1, 2, 3, 11, 17, 23, 29, 35],
        use_rope=False, use_attn_gate=True,
    ),
    # All three flags at AFM's defaults -> bit-identical to the teacher.
    # Only useful if you want A3 runnable from disk; it costs ~10 GB.
    "baseline": dict(
        dirname="afm_baseline_init",
        full_attn=list(range(N_LAYERS)),
        use_rope=True, use_attn_gate=False,
    ),
}


def git_info(path: str) -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", path, *args],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    commit = run("rev-parse", "HEAD")
    return {
        "commit": commit,
        "short": commit[:8] if commit else None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def pkg_version(name: str):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="hybrid31", choices=sorted(VARIANTS))
    ap.add_argument("--out", default=None,
                    help="output directory (default: $WORK_LUKAS/students/<variant dirname>)")
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--full-attn", default=None,
                    help="comma-separated 0-based full-attention layer indices; overrides the variant")
    ap.add_argument("--no-transfer", action="store_true",
                    help="skip the teacher q/k/v/o transfer into KDA (RADLADS de-novo ablation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite an existing output directory")
    ap.add_argument("--dry-run", action="store_true", help="build and check, but do not save")
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    torch.manual_seed(args.seed)

    full_attn = ([int(x) for x in args.full_attn.split(",")] if args.full_attn
                 else list(spec["full_attn"]))
    full_attn = sorted(set(full_attn))
    assert all(0 <= i < N_LAYERS for i in full_attn), full_attn
    kda_layers = [i for i in range(N_LAYERS) if i not in full_attn]

    if args.variant == "hybrid31" and args.full_attn is None:
        assert kda_layers == build_kda_layers(N_LAYERS, 4), "hybrid31 layout drifted"

    out = args.out or f"{WORK}/students/{spec['dirname']}"
    if args.no_transfer and args.out is None:
        out = out.replace("_init", "_denovo_init")
    if os.path.exists(out) and not (args.force or args.dry_run):
        sys.exit(f"{out} already exists -- pass --force to overwrite")

    n_kda, n_full = len(kda_layers), len(full_attn)
    print(f"variant       : {args.variant}")
    print(f"KDA layers    : {n_kda}   full attention: {n_full} at {full_attn}")
    print(f"use_rope      : {spec['use_rope']}   use_attn_gate: {spec['use_attn_gate']}")
    print(f"transfer q/k/v/o : {not args.no_transfer}")
    print(f"output        : {out}\n")

    # --- teacher ----------------------------------------------------------
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"teacher parameters: {teacher_params:,}")

    # --- student ----------------------------------------------------------
    cfg = AFMKDAConfig.from_pretrained(
        args.teacher,
        kda_layers=kda_layers,                 # 0-BASED
        use_rope=spec["use_rope"],
        use_attn_gate=spec["use_attn_gate"],
    )
    # These two are the A8 bugs; assert rather than trust.
    assert cfg.model_type == "afm_kda", cfg.model_type
    assert cfg.auto_map and "AutoModelForCausalLM" in cfg.auto_map, cfg.auto_map
    assert cfg.vocab_size == 128004, cfg.vocab_size

    student, info = AFMKDAForCausalLM.from_pretrained(
        args.teacher, config=cfg, dtype=torch.bfloat16, output_loading_info=True)

    missing = list(info.get("missing_keys", []))
    unexpected = list(info.get("unexpected_keys", []))
    print(f"missing keys: {len(missing)}   unexpected keys: {len(unexpected)}"
          f"   (both non-zero is EXPECTED whenever n_kda > 0)")

    init_new_parameters(student, missing)      # fills ONLY the new parameters
    if not args.no_transfer:
        transfer_attention_into_kda(student, teacher)

    # The teacher is ~9.2 GB and is no longer needed -- drop peak RAM before saving.
    del teacher
    gc.collect()

    # --- checks -----------------------------------------------------------
    n = sum(p.numel() for p in student.parameters())
    expected = (teacher_params + n_kda * KDA_DELTA
                + (n_full * GATE_PARAMS if spec["use_attn_gate"] else 0))
    print(f"parameters: {n:,}   expected {expected:,}   ({n/1e9:.6f} B)")
    assert n == expected, f"parameter count off by {n - expected:+,}"

    bad = [k for k, p in student.named_parameters() if not torch.isfinite(p).all()]
    assert not bad, f"non-finite parameters: {bad[:5]}"

    kinds = [type(l.self_attn).__name__ for l in student.model.layers]
    assert kinds.count("AFMKDALinearAttention") == n_kda, kinds
    print(f"layer types: {kinds.count('AFMKDALinearAttention')} KDA, "
          f"{kinds.count('AFMKDAAttention')} attention")

    if args.dry_run:
        print("\n--dry-run: nothing saved")
        return

    # --- save -------------------------------------------------------------
    os.makedirs(out, exist_ok=True)
    student.save_pretrained(out)
    AutoTokenizer.from_pretrained(args.teacher).save_pretrained(out)

    for f in CODE_FILES:
        shutil.copy(f"{SRC}/{f}", out)

    info_json = {
        "variant": args.variant,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "teacher": args.teacher,
        "teacher_parameters": teacher_params,
        "parameters": n,
        "config": {
            "kda_layers": kda_layers,
            "full_attn_layers": full_attn,
            "num_kda_layers": n_kda,
            "use_rope": spec["use_rope"],
            "use_attn_gate": spec["use_attn_gate"],
        },
        "attention_transferred": not args.no_transfer,
        "seed": args.seed,
        "loading_info": {"missing_keys": len(missing), "unexpected_keys": len(unexpected)},
        "afm_kda_git": git_info(SRC),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "flash-linear-attention": pkg_version("flash-linear-attention"),
            "fla-core": pkg_version("fla-core"),
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "code_files": list(CODE_FILES),
    }
    with open(f"{out}/build_info.json", "w") as fh:
        json.dump(info_json, fh, indent=2)

    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)
               if os.path.isfile(os.path.join(out, f)))
    print(f"\nsaved to {out}  ({size/2**30:.1f} GiB)")
    if info_json["afm_kda_git"]["dirty"]:
        print("WARNING: afm_kda has uncommitted changes -- build_info.json's hash is not the code you ran")
    print("\nIf you rebuild or rename, clear the stale module cache first:")
    print(f"  rm -rf $HF_HOME/modules/transformers_modules/{os.path.basename(out)}")


if __name__ == "__main__":
    main()