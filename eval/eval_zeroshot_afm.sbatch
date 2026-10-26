#!/bin/bash
#SBATCH --account=EUHPC_D34_049
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:29:59
#SBATCH -p boost_usr_prod
#SBATCH -J eval_zs_r3_stage2_kda_only_finemath
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/eval_logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/eval_logs/%x-%j.err

# =====================================================================
# ZERO-SHOT REGIME -- ARC-Challenge, HellaSwag, WinoGrande.
# Metrics: acc_norm (ARC-C, HellaSwag), acc (WinoGrande).
#
# THE PROTOCOL IS FROZEN.  It matches RADLADS (all 0-shot except 5-shot
# MMLU) and your own practical work, so teacher, students and the paper
# are all on one axis.  Change only NAME and MODEL_PATH per model.
#
# NO --apply_chat_template: AFM-4.5B-Base and every student are base
# models, scored in native completion format.
# =====================================================================

source /leonardo/home/userexternal/lrosenbe/eval-env2/bin/activate

export HF_HOME=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache
# NOTE: must be $HF_HOME/hub, NOT $HF_HOME -- setting it one level too high makes
# every offline repo-id load fail with a misleading *network* error.
export HUGGINGFACE_HUB_CACHE=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

# Explicit snapshot path, not the repo id: it works offline regardless of cache
# env vars, AND the hash is a provenance record pinning the exact bytes measured.


#NAME=teacher_afm45b_base
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache/hub/models--arcee-ai--AFM-4.5B-Base/snapshots/dab7922f9c868d479b365410304f466b007b1c5a

#NAME=r1_stage1_hybrid31
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r1_stage1_hybrid31/checkpoint-4580

#NAME=r2_stage1_gdn_hybrid31
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r2_stage1_gdn_hybrid31/

#NAME=r3_stage1_kda_only
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r3_stage1_kda_only/

#NAME=r4_stage1_gdn_only
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r4_stage1_gdn_only/

#NAME=r5_stage1_kda_hybrid31_1ep
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r5_stage1_kda_hybrid31_1ep

#NAME=r1_stage2_kda_hybrid31_unfreeze
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r1_stage2_kda_hybrid31_unfreeze

#NAME=r2_stage2_gdn_hybrid31_unfreeze
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r2_stage2_gdn_hybrid31_unfreeze

#NAME=r3_stage2_kda_only_unfreeze
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r3_stage2_kda_only_unfreeze

#NAME=r4_stage2_gdn_only_unfreeze
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r4_stage2_gdn_only_unfreeze

#NAME=r1_stage2_kda_hybrid31_cosine5e5
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r1_stage2_kda_hybrid31_cosine5e5

#NAME=r1_stage1_hybrid31_s43
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r1_stage1_hybrid31_s43

#NAME=r1_stage1_hybrid31_s44
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r1_stage1_hybrid31_s44

#NAME=r2_stage1_gdn_hybrid31_s43
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r2_stage1_gdn_hybrid31_s43

#NAME=r2_stage1_gdn_hybrid31_s44
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r2_stage1_gdn_hybrid31_s44

#NAME=r1_stage1_hybrid31_finemath
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r1_stage1_hybrid31_finemath

#NAME=r1_stage2_kda_hybrid31_finemath
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r1_stage2_kda_hybrid31_finemath

#NAME=r1fm_stage2_kda_hybrid31_finemath
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r1fm_stage2_kda_hybrid31_finemath

#NAME=r2_stage2_gdn_hybrid31_finemath
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r2_stage2_gdn_hybrid31_finemath

#NAME=r4_stage2_gdn_only_finemath
#MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r4_stage2_gdn_only_finemath

NAME=r3_stage2_kda_only_finemath
MODEL_PATH=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/output/r3_stage2_kda_only_finemath


echo "=== provenance ==="
python -c "import lm_eval, transformers, torch; print('lm_eval', lm_eval.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__)"
echo "model=$MODEL_PATH   name=$NAME"
echo "started $(date)"

srun lm_eval \
  --model hf \
  --model_args pretrained=$MODEL_PATH,dtype=bfloat16,trust_remote_code=True,use_cache=True \
  --tasks arc_challenge,hellaswag,winogrande \
  --num_fewshot 0 \
  --batch_size auto \
  --seed 1234 \
  --output_path /leonardo_work/EUHPC_D34_049/WORK_LUKAS/eval_results/${NAME}_zeroshot_cache

echo "finished $(date)"