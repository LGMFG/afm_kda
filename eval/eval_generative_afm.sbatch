#!/bin/bash
#SBATCH --account=EUHPC_D34_049
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --qos=normal
#SBATCH --time=14:30:00
#SBATCH -p boost_usr_prod
#SBATCH -J eval_gsm8k_r5_stage1_kda_hybrid31_1ep
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/eval_logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/eval_logs/%x-%j.err

# =====================================================================
# B8 -- THE GENERATIVE REGIME.  GSM8K and IFEval.
#
# These are the first benchmarks that actually WRITE text, so they are
# the first that depend on Track B.  The four likelihood-scored tasks
# never needed it.
#
# SHOT COUNTS -- frozen here, do not change once a student is measured:
#   gsm8k   5-shot   (lm-eval's task default).  NOT 0-shot: a base model
#                    at 0-shot emits prose instead of an answer and both
#                    teacher and student score ~0, which measures nothing.
#   ifeval  0-shot   (its only mode -- the task IS the instruction).
#   NOT the CoT variants.
#
# NO --apply_chat_template: AFM-4.5B-Base and every student are base models.
#
# ⚠️ use_cache=True IS PASSED EXPLICITLY, FOR BOTH MODELS.
#    B7c measured the teacher loading with config.use_cache=False and the
#    student with True -- Arcee ship their configs that way, and our
#    finalize_checkpoint.py flips only ours.  Left alone, the teacher would
#    run O(n^2) and the student O(n), and any timing comparison between them
#    would be meaningless.  Passing it here puts both on the same footing
#    whatever their config files say, and does not touch the checkpoints.
# =====================================================================

source /leonardo/home/userexternal/lrosenbe/eval-env2/bin/activate

export HF_HOME=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache
# must be $HF_HOME/hub, NOT $HF_HOME -- one level too high and every offline
# repo-id load fails with a misleading *network* error.
export HUGGINGFACE_HUB_CACHE=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

W=/leonardo_work/EUHPC_D34_049/WORK_LUKAS
TEACHER=$W/hf_cache/hub/models--arcee-ai--AFM-4.5B-Base/snapshots/dab7922f9c868d479b365410304f466b007b1c5a
STUDENT1=$W/output/r1_stage1_hybrid31/checkpoint-4580
STUDENT2=$W/output/r2_stage1_gdn_hybrid31
STUDENT3=$W/output/r3_stage1_kda_only
STUDENT4=$W/output/r4_stage1_gdn_only
STUDENT5=$W/output/r5_stage1_kda_hybrid31_1ep

# ${VAR:-default} so --export wins.  A plain assignment would silently beat it.
NAME=${NAME:-r5_stage1_kda_hybrid31_1ep}
MODEL_PATH=${MODEL_PATH:-$STUDENT5}
TASKS=${TASKS:-gsm8k}
NSHOT=${NSHOT:-5}
BATCH=${BATCH:-auto}
LIMIT=${LIMIT:-}          # set LIMIT=20 for a debug-queue smoke run

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

# Samples are ~10 KB/item -- 13 MB for all of GSM8K.  They are the only
# record of HOW the model failed, and a score cannot tell you that a
# generation never terminated.  Log by default; opt out explicitly.
LOG_ARG="--log_samples"
[ "${NO_LOG_SAMPLES:-0}" = "1" ] && LOG_ARG=""

mkdir -p $W/eval_logs $W/eval_results

echo "=== provenance ==="
python -c "import lm_eval, transformers, torch; print('lm_eval', lm_eval.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__)"
python -c "from importlib.metadata import version; print('fla', version('flash-linear-attention'))" 2>/dev/null || echo "fla: NOT INSTALLED (fine for the teacher, fatal for a student)"
echo "name=$NAME  tasks=$TASKS  nshot=$NSHOT  limit=${LIMIT:-none}"
echo "model=$MODEL_PATH"
[ -f "$MODEL_PATH/afm_kda_commit.txt" ] && echo "afm_kda commit: $(cat $MODEL_PATH/afm_kda_commit.txt)"
echo "started $(date)"

srun lm_eval \
  --model hf \
  --model_args pretrained=$MODEL_PATH,dtype=bfloat16,trust_remote_code=True,use_cache=True \
  --tasks $TASKS \
  --num_fewshot $NSHOT \
  --batch_size $BATCH \
  --seed 1234 \
  $LOG_ARG \
  $LIMIT_ARG \
  --output_path $W/eval_results/${NAME}_${TASKS}_${NSHOT}shot

echo "finished $(date)"

# =====================================================================
# HOW TO RUN IT -- smoke first, on the debug queue.  Four submissions.
#
#   S=$W/afm_kda/eval_generative_afm.sbatch
#   ST=$W/output/r1_stage1_hybrid31
#
#   # 1. smoke, ~20 items each, 30 min debug qos -- proves the pipeline
#   sbatch --export=ALL,QOS=boost_qos_dbg,LIMIT=20,NAME=smoke_teacher $S
#   sbatch --export=ALL,QOS=boost_qos_dbg,LIMIT=20,NAME=smoke_student,MODEL_PATH=$ST $S
#
#   # 2. the real runs, one task per job so a timeout loses only one number
#   sbatch --export=ALL,NAME=teacher_afm45b_base $S
#   sbatch --export=ALL,NAME=r1_stage1_hybrid31,MODEL_PATH=$ST $S
#   sbatch --export=ALL,NAME=teacher_afm45b_base,TASKS=ifeval,NSHOT=0 $S
#   sbatch --export=ALL,NAME=r1_stage1_hybrid31,MODEL_PATH=$ST,TASKS=ifeval,NSHOT=0 $S
#
# ⚠️ --time is a #SBATCH directive and is NOT overridable by --export.
#    If a job times out, raise it in the file or split with --limit.
# ⚠️ batch_size=auto probes with a forward pass; if it OOMs during
#    generation, pass BATCH=8 explicitly.
# ⚠️ eval-env2 needs flash-linear-attention for ANY student run.  The
#    provenance block above prints it -- read that line before assuming
#    a crash is our code.
# =====================================================================