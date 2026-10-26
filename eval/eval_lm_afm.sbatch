#!/bin/bash
#SBATCH --account=EUHPC_D34_049
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=08:00:00
#SBATCH -p boost_usr_prod
#SBATCH -J eval_lm
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/eval_logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/eval_logs/%x-%j.err

# =====================================================================
# ONE lm-eval script for the GENERATIVE benchmarks that go to the lprod queue.   (v3, 2026-10-16)
# The 0-shot trio and MMLU are run by hand on the debug queue (eval_zeroshot_afm /
# eval_mmlu_afm.sbatch): the debug QoS cancels any job beyond two, so nothing is queued there.
#   REGIME=gsm8k      gsm8k       5-shot                 (strict exact match)
#   REGIME=humaneval  humaneval   TASK DEFAULT (0-shot)  (pass@1, greedy)       NEW
#   REGIME=mbpp       mbpp        TASK DEFAULT (3-shot)  (pass@1, greedy)       NEW
# GSM8K's shot count is the frozen protocol every earlier number used, and the output
# directory name is the old one, so old and new results sit side by side.
# HumanEval / MBPP: NO --num_fewshot is passed -- the shot count is part of the
# task definition in lm-eval (HumanEval: 0, the Codex protocol; MBPP: the 3 fixed
# prompt examples of Austin et al. 2021, task_ids 2-4).  The job prints the task
# yaml so the protocol actually used is in the log, and results_*.json records n-shot.
#
# HUMANEVAL vs the practical work's humaneval_instruct:
#   humaneval_instruct wraps each problem in a chat-style instruction and was
#   run with --apply_chat_template -- right for the instruct-distilled Qwen
#   students, wrong here.  AFM-4.5B-Base and every student are BASE models:
#   plain `humaneval` gives the function signature + docstring and the model
#   completes the body (the original Codex protocol, and what Arcee/RADLADS-
#   style base-model tables report).  No chat template, 0-shot, greedy, the
#   task's own stop sequences, max_gen_toks from the task yaml.
#
# Changes against the old per-regime scripts:
#   * --cpus-per-task 32, as in every earlier eval run: same environment as the
#     numbers it is compared with.  (Billed as a full node; 8 would be 4x cheaper
#     and is no slower -- the work is on the GPU -- but cost is not the constraint.)
#   * --log_samples ALWAYS (rev 31 trap: samples welded to --limit lost every
#     full GSM8K generation).  Per-item samples are what make PAIRED tests
#     (stage 1 vs stage 2 on the same questions) possible -- far tighter than
#     the unpaired CIs used so far.
#   * use_cache=True for every regime and every model (does not change
#     likelihood scores; makes generation O(n) for teacher and students alike).
#
# Variables are read as ${VAR:-default} so --export wins.
# --time / --qos: pass them on the sbatch COMMAND LINE (sbatch --time=01:00:00 ...),
# which overrides the #SBATCH lines above; --export cannot.  queue_evals.sh does this.
# =====================================================================

source /leonardo/home/userexternal/lrosenbe/eval-env2/bin/activate

W=/leonardo_work/EUHPC_D34_049/WORK_LUKAS
export HF_HOME=$W/hf_cache
export HF_DATASETS_CACHE=$W/hf_cache
export HUGGINGFACE_HUB_CACHE=$W/hf_cache/hub     # $HF_HOME/hub, NOT $HF_HOME
export HF_MODULES_CACHE=$HF_HOME/modules
export EVALUATE_MODULES_CACHE=$HF_HOME/modules
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_ALLOW_CODE_EVAL=1                     # code_eval refuses to execute without it
export TOKENIZERS_PARALLELISM=false

TEACHER=$W/hf_cache/hub/models--arcee-ai--AFM-4.5B-Base/snapshots/dab7922f9c868d479b365410304f466b007b1c5a

REGIME=${REGIME:?set REGIME=gsm8k|humaneval|mbpp}
NAME=${NAME:?set NAME, e.g. r1_stage2_kda_hybrid31_unfreeze_ck15260}
MODEL_PATH=${MODEL_PATH:-$TEACHER}
BATCH=${BATCH:-auto}
LIMIT=${LIMIT:-}                                # LIMIT=16 for a smoke run (then NAME=smoke_...)

UNSAFE=""
case "$REGIME" in
  gsm8k)     TASKS=gsm8k;     NSHOT=5;  OUT=${NAME}_gsm8k_5shot ;;
  humaneval) TASKS=humaneval; NSHOT=""; OUT=${NAME}_humaneval; UNSAFE=--confirm_run_unsafe_code ;;
  mbpp)      TASKS=mbpp;      NSHOT=""; OUT=${NAME}_mbpp;      UNSAFE=--confirm_run_unsafe_code ;;
  *) echo "FATAL: unknown REGIME=$REGIME"; exit 1 ;;
esac
OUTDIR=$W/eval_results/$OUT

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
NSHOT_ARG=""
[ -n "$NSHOT" ] && NSHOT_ARG="--num_fewshot $NSHOT"

# --- what this job needs to load the model at all.  finalize_checkpoint.py is NOT
# required: use_cache=True is passed explicitly below, which is all its flip did for
# lm-eval.  The real requirement is that the checkpoint carries the model code
# (trust_remote_code) -- queue_evals.sh checks that at submission time already;
# this repeats it so a hand-submitted job fails in seconds with a clear message.
[ -f "$MODEL_PATH/config.json" ] || { echo "FATAL: no config.json in $MODEL_PATH"; exit 1; }
if [ "$MODEL_PATH" != "$TEACHER" ]; then
  [ -f "$MODEL_PATH/modeling_afm_kda.py" ] || { echo "FATAL: $MODEL_PATH has no modeling_afm_kda.py -- cannot load with trust_remote_code"; exit 1; }
fi
if ls "$OUTDIR"/*/results_*.json >/dev/null 2>&1 || ls "$OUTDIR"/results_*.json >/dev/null 2>&1; then
  echo "SKIP: results already exist in $OUTDIR"; exit 0
fi

mkdir -p $W/eval_logs $W/eval_results

echo "=== provenance ==="
python -c "import lm_eval, transformers, torch; print('lm_eval', lm_eval.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__)"
python -c "from importlib.metadata import version; print('fla', version('flash-linear-attention'))" 2>/dev/null || echo "fla: NOT INSTALLED (fine for the teacher, fatal for a student)"
echo "regime=$REGIME  tasks=$TASKS  nshot=${NSHOT:-task default}  limit=${LIMIT:-none}  batch=$BATCH"
if [ -n "$UNSAFE" ]; then   # code tasks: print the task definition = the protocol actually used
  python - "$TASKS" <<'PY'
import glob, os, sys, lm_eval
t = sys.argv[1]
for f in glob.glob(os.path.join(os.path.dirname(lm_eval.__file__), "tasks", "**", f"{t}.yaml"), recursive=True):
    print(f"--- task yaml: {f}"); print(open(f).read())
PY
fi
echo "name=$NAME"
echo "model=$MODEL_PATH"
[ -f "$MODEL_PATH/afm_kda_commit.txt" ] && echo "afm_kda commit: $(cat $MODEL_PATH/afm_kda_commit.txt)"
[ -f "$MODEL_PATH/trainer_state.json" ] && python -c "import json;print('global_step', json.load(open('$MODEL_PATH/trainer_state.json'))['global_step'])"
echo "started $(date)"

srun lm_eval \
  --model hf \
  --model_args pretrained=$MODEL_PATH,dtype=bfloat16,trust_remote_code=True,use_cache=True \
  --tasks $TASKS \
  $NSHOT_ARG \
  --batch_size $BATCH \
  --seed 1234 \
  $UNSAFE \
  $LIMIT_ARG \
  --log_samples \
  --output_path $OUTDIR
RC=$?
echo "lm_eval exited $RC   finished $(date)"
exit $RC

# =====================================================================
# SMOKE FIRST (debug queue, 10 problems): proves the dataset is in the
# offline cache, code execution is allowed, and a student generates code.
#   S=$W/afm_kda/eval_lm_afm.sbatch
#   sbatch --qos=boost_qos_dbg --time=00:30:00 --export=ALL,REGIME=humaneval,LIMIT=16,NAME=smoke_teacher $S
#   sbatch --qos=boost_qos_dbg --time=00:30:00 --export=ALL,REGIME=humaneval,LIMIT=16,NAME=smoke_r1,MODEL_PATH=$W/output/r1_stage1_hybrid31/checkpoint-4580 $S
#   (same with REGIME=mbpp, LIMIT=50)
# Read one generated completion in the samples file before queueing the rest.
# TIMING: (finished - started) of the smoke x 164/16 (HumanEval) or x 500/50 (MBPP)
#   minus model loading = the full run; queue_evals.sh asks for 8 h / 12 h anyway.
# =====================================================================