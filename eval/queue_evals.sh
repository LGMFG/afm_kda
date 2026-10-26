#!/bin/bash
# =====================================================================
# Queue GSM8K, HumanEval and MBPP on boost_qos_lprod; print the manual debug
# commands for the arithmetic probe and ce_outliers.                 (v4, 2026-10-16)
#
#   DRY=1 bash queue_evals.sh all      print the sbatch lines only -- do this first
#   bash queue_evals.sh all            submit everything below: 31 jobs (21 GSM8K/HumanEval/MBPP on 5 finals + 4 seeds + 2 ckpts, 10 code on teacher + R1-R4)
#   bash queue_evals.sh debug          DOES NOT SUBMIT: writes the arithmetic model lists and
#                                      prints the sbatch commands to run by hand on debug
#
# PLANs (all = stage2 s2ckpt seeds code1):
#   stage2  5 stage-2 finals (R1-S2..R4-S2, R1-S2-cos @ ck 15260): GSM8K, HumanEval, MBPP
#   s2ckpt  R1-S2 at 300 M (ck 4578) and 500 M (ck 7630):          GSM8K
#   seeds   R1/R2 x seed 43/44 (stage 1, ck 4580):                 GSM8K
#   code1   teacher + R1-R4 stage 1:                                HumanEval, MBPP
#   debug   (no submission) arithmetic lists + commands; ce_outliers commands
#
# NOTHING is submitted to the debug queue (it cancels every job beyond two per user).
# The 0-shot trio, MMLU, the arithmetic probe and ce_outliers are run by hand there.
#
# Before submitting, every model directory is checked here, on the login node:
# config.json, modeling_afm_kda.py (trust_remote_code needs it) and tokenizer files.
# A failing directory is skipped with a message instead of failing after days in the
# queue.  finalize_checkpoint.py is not needed: use_cache=True is passed explicitly.
# Every lm-eval job skips itself if its results already exist, so re-running a PLAN
# only resubmits what is missing.
# =====================================================================
set -u
W=/leonardo_work/EUHPC_D34_049/WORK_LUKAS
O=$W/output
A=$W/afm_kda
S=$W/eval_lm_afm.sbatch
TEACHER=$W/hf_cache/hub/models--arcee-ai--AFM-4.5B-Base/snapshots/dab7922f9c868d479b365410304f466b007b1c5a
DRY=${DRY:-}
QOS=boost_qos_lprod
# walltimes: generous on purpose ("better safe than sorry"); a long request may wait
# a little longer for a backfill slot, but a timeout costs a whole queue wait.
declare -A TIME=( [gsm8k]=06:00:00 [humaneval]=08:00:00 [mbpp]=12:00:00 )

# ---- the models -------------------------------------------------------
S2=( "r1_s2_kda_hyb=$O/r1_stage2_kda_hybrid31_unfreeze/checkpoint-15260"
     "r2_s2_gdn_hyb=$O/r2_stage2_gdn_hybrid31_unfreeze/checkpoint-15260"
     "r3_s2_kda_only=$O/r3_stage2_kda_only_unfreeze/checkpoint-15260"
     "r4_s2_gdn_only=$O/r4_stage2_gdn_only_unfreeze/checkpoint-15260"
     "r1_s2cos_kda_hyb=$O/r1_stage2_kda_hybrid31_cosine5e5/checkpoint-15260" )
S2CK=( "r1_s2_kda_hyb_ck04578=$O/r1_stage2_kda_hybrid31_unfreeze/checkpoint-4578"
       "r1_s2_kda_hyb_ck07630=$O/r1_stage2_kda_hybrid31_unfreeze/checkpoint-7630" )
SEEDS=( "r1_s1_kda_hyb_seed43=$O/r1_stage1_hybrid31_s43/checkpoint-4580"
        "r1_s1_kda_hyb_seed44=$O/r1_stage1_hybrid31_s44/checkpoint-4580"
        "r2_s1_gdn_hyb_seed43=$O/r2_stage1_gdn_hybrid31_s43/checkpoint-4580"
        "r2_s1_gdn_hyb_seed44=$O/r2_stage1_gdn_hybrid31_s44/checkpoint-4580" )
# names of the stage-1 models = the names their earlier results were written under
S1=( "teacher_afm45b_base=$TEACHER"
     "r1_stage1_hybrid31=$O/r1_stage1_hybrid31/checkpoint-4580"
     "r2_stage1_gdn_hybrid31=$O/r2_stage1_gdn_hybrid31/checkpoint-4580"
     "r3_stage1_kda_only=$O/r3_stage1_kda_only/checkpoint-4580"
     "r4_stage1_gdn_only=$O/r4_stage1_gdn_only/checkpoint-4580" )

N=0; BAD=0
run() { echo "+ $*"; if [ -z "$DRY" ]; then "$@" || echo "!! sbatch FAILED (see message above)"; fi; N=$((N+1)); }

loadable() {   # loadable PATH -> 0 if the directory can be loaded by lm-eval / the probe
  local p=$1
  [ -d "$p" ]                  || { echo "!! MISSING $p -- skipped"; BAD=$((BAD+1)); return 1; }
  [ -f "$p/config.json" ]      || { echo "!! no config.json in $p -- skipped"; BAD=$((BAD+1)); return 1; }
  [ "$p" = "$TEACHER" ] && return 0
  [ -f "$p/modeling_afm_kda.py" ] || { echo "!! no modeling_afm_kda.py in $p -- copy the code files in (finalize_checkpoint.py --sync) -- skipped"; BAD=$((BAD+1)); return 1; }
  ls "$p"/tokenizer*.json >/dev/null 2>&1 || { echo "!! no tokenizer files in $p -- skipped"; BAD=$((BAD+1)); return 1; }
  return 0
}

lm() {   # lm REGIME name=path ...
  local regime=$1; shift
  for m in "$@"; do
    local name=${m%%=*} path=${m#*=}
    loadable "$path" || continue
    run sbatch --qos=$QOS --time=${TIME[$regime]} -J e_${regime}_${name} \
        --export=ALL,REGIME=$regime,NAME=$name,MODEL_PATH=$path $S
  done
}

arith_list() {   # arith_list TAG name=path ...  -> list file + the debug command to run by hand
  local tag=$1; shift
  local ok=() m
  for m in "$@"; do loadable "${m#*=}" && ok+=("$m"); done
  [ ${#ok[@]} -gt 0 ] || return 0
  local f=$W/eval_logs/arith_lists/$tag.txt
  mkdir -p "$(dirname "$f")"
  printf '%s\n' "${ok[@]}" > "$f"
  echo "  # $tag: ${#ok[@]} models -> $f"
  echo "  sbatch -J arith_$tag --export=ALL,SET=custom,MODELS_FILE=$f $A/arith_probe.sbatch"
}

[ $# -gt 0 ] || { echo "usage: [DRY=1] bash queue_evals.sh all|stage2|s2ckpt|seeds|code1|debug ..."; exit 1; }
PLANS=()
for p in "$@"; do [ "$p" = all ] && PLANS+=(stage2 s2ckpt seeds code1) || PLANS+=("$p"); done

for plan in "${PLANS[@]}"; do
  echo "=== $plan"
  case "$plan" in
    stage2) lm gsm8k "${S2[@]}"; lm humaneval "${S2[@]}"; lm mbpp "${S2[@]}" ;;
    s2ckpt) lm gsm8k "${S2CK[@]}" ;;
    seeds)  lm gsm8k "${SEEDS[@]}" ;;
    code1)  lm humaneval "${S1[@]}"; lm mbpp "${S1[@]}" ;;
    debug)
      echo "--- ARITHMETIC PROBE: run these by hand, at most TWO at a time (debug, 30 min)."
      echo "    A job killed at 30 min is finished by resubmitting the SAME line: models whose"
      echo "    <name>.summary.json exists in eval_results/arith_probe_v2 are skipped."
      arith_list s2final "${S2[@]}"
      C=(); for s in 1526 3052 4578 6104 7630; do C+=("r1_s2_ck$(printf %05d $s)=$O/r1_stage2_kda_hybrid31_unfreeze/checkpoint-$s"); done
      arith_list r1s2curve1 "${C[@]}"
      C=(); for s in 9156 10682 12208 13734; do C+=("r1_s2_ck$(printf %05d $s)=$O/r1_stage2_kda_hybrid31_unfreeze/checkpoint-$s"); done
      arith_list r1s2curve2 "${C[@]}"     # ck 15260 = r1_s2_kda_hyb in s2final
      arith_list seeds "${SEEDS[@]}"
      echo "--- CE_OUTLIERS: one model per job (the script calls it 'R5'; it is compared with R1 stage 1)."
      for m in "${S2[@]}"; do
        loadable "${m#*=}" || continue
        echo "  sbatch -J outl_${m%%=*} --export=ALL,R5=${m#*=} $A/ce_outliers.sbatch"
      done
      echo "--- then:  cd $A && python arith_stage2.py $W/eval_results/arith_probe_v2" ;;
    *) echo "unknown PLAN $plan"; exit 1 ;;
  esac
done
echo "=== $N submissions${DRY:+ (dry run -- nothing submitted)}, $BAD directories skipped"