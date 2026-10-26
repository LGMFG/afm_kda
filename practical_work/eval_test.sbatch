#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH -p boost_usr_prod
#SBATCH -J 10k_arcc_hs_wg
#SBATCH -o eval_logs/%x-%j.out
#SBATCH -e eval_logs/%x-%j.err

# Activate your env
source /leonardo/home/userexternal/lrosenbe/eval-env2/bin/activate

# CRITICAL: HuggingFace offline setup
export HF_HOME=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache
#export TRANSFORMERS_CACHE=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache
export HUGGINGFACE_HUB_CACHE=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

ls -lah $HF_DATASETS_CACHE | head
find $HF_DATASETS_CACHE -maxdepth 2 | grep ai2_arc


echo "Starting at $(date)"
nvidia-smi


# Run lm-eval-harness with the specified config file
srun lm_eval \
  --model hf \
  --model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_10k_4B_instr_06B/checkpoint-226,dtype=bfloat16 \
  #--tasks arc_easy \
  --tasks arc_challenge,hellaswag,winogrande
  #--tasks gsm8k,mmlu \
  --apply_chat_template \
  --fewshot_as_multiturn \
  #--device cuda:0 \
  --batch_size auto \
  #--num_fewshot 5 \
  --num_fewshot 0 \
  --output_path ./eval_results/test10k_arcc_hs_wg

echo "Finished at $(date)"