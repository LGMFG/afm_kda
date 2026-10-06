#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J arith_probe
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/afm_kda/logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/afm_kda/logs/%x-%j.err

# =====================================================================
# Arithmetic probe v2: copy control + digit-length sweep, all models.
#
#   sbatch --export=ALL,SET=a      arith_probe.sbatch    teacher, R1, R2
#   sbatch --export=ALL,SET=b      arith_probe.sbatch    R3, R4, R5 (final)
#   sbatch --export=ALL,SET=curve1 arith_probe.sbatch    R5 checkpoints 1-5
#   sbatch --export=ALL,SET=curve2 arith_probe.sbatch    R5 checkpoints 6-10
#   sbatch --export=ALL,SET=b1     arith_probe.sbatch    R1 + R5 at batch size 1 (control)
#
# Debug QoS runs two at once: submit a + b together, then curve1 + curve2.
# A job killed at the 30-minute limit is FINISHED by resubmitting the same
# command -- any model whose <name>.summary.json exists is skipped.
#
# SET is read with ${SET:?}, never assigned here, because a plain
# assignment in the script would silently beat --export.
# =====================================================================
: "${SET:?submit with --export=ALL,SET=a|b|curve1|curve2|b1}"

source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate
export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS
export HF_HOME=$WORK_LUKAS/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

PROBE_DIR=$WORK_LUKAS/afm_kda
OUT=$WORK_LUKAS/eval_results/arith_probe_v2
O=$WORK_LUKAS/output
TEACHER=$WORK_LUKAS/hf_cache/hub/models--arcee-ai--AFM-4.5B-Base/snapshots/dab7922f9c868d479b365410304f466b007b1c5a
R5=$O/r5_stage1_kda_hybrid31_1ep

# ---- R2-R4: the three lines marked CHECK must name your real directories.
# ---- `ls -d $WORK_LUKAS/output/*/checkpoint-4580` lists them.  A wrong
# ---- path is caught before any model loads, so it costs seconds, not GPU.
case "$SET" in
  a) MODELS=(
       "teacher=$TEACHER"
       "r1_kda_hybrid=$O/r1_stage1_hybrid31/checkpoint-4580"   # top-level was overwritten 09-29
       "r2_gdn_hybrid=$O/r2_stage1_gdn_hybrid31/checkpoint-4580"            # <-- CHECK
     ) ;;
  b) MODELS=(
       "r3_kda_only=$O/r3_stage1_kda_only/checkpoint-4580"                  # <-- CHECK
       "r4_gdn_only=$O/r4_stage1_gdn_only/checkpoint-4580"                  # <-- CHECK
       "r5_kda_hybrid_1ep=$R5"                     # top-level = the model R5's 0-shot eval used
     ) ;;
  curve1) MODELS=()
     for s in 3494 6988 10482 13976 17470; do
       MODELS+=("r5_ck$(printf %05d $s)=$R5/checkpoint-$s"); done ;;
  curve2) MODELS=()
     # checkpoint-34940 is the same weights as R5's top-level: a free identity check
     for s in 20964 24458 27952 31446 34940; do
       MODELS+=("r5_ck$(printf %05d $s)=$R5/checkpoint-$s"); done ;;
  b1) # BATCHING CONTROL: R1 and R5 at batch size 1 into a separate directory.
      # Compared problem-by-problem with the batched run, this measures how
      # much batching moved any ANSWER (the built-in check compares whole
      # 12-token continuations, most of which lie after the answer).
     OUT=$WORK_LUKAS/eval_results/arith_probe_v2_bs1
     EXTRA="--batch-size 1 --no-check-batching"
     MODELS=(
       "r1_kda_hybrid=$O/r1_stage1_hybrid31/checkpoint-4580"
       "r5_kda_hybrid_1ep=$R5"
     ) ;;
  *) echo "FATAL: unknown SET=$SET"; exit 1 ;;
esac

ARGS=()
for m in "${MODELS[@]}"; do ARGS+=(--model "$m"); done

echo "=== SET=$SET   ${#MODELS[@]} models   -> $OUT"
printf '    %s\n' "${MODELS[@]}"
python -c "import torch, transformers; from importlib.metadata import version; \
print('VERSIONS torch', torch.__version__, '| transformers', transformers.__version__, \
'| fla', version('flash-linear-attention'))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python $PROBE_DIR/arith_probe.py --out $OUT --tokenizer $TEACHER ${EXTRA:-} "${ARGS[@]}"
RC=$?
echo "arith_probe exited $RC"
exit $RC