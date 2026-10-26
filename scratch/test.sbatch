#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -p boost_usr_prod
#SBATCH -o gpu_out.txt
#SBATCH -e gpu_err.txt

# Activate your env
source /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distill_env/bin/activate


export HF_HOME=$WORK/WORK_LUKAS/hf_cache # Set Hugging Face cache directory
export HF_HUB_OFFLINE=1 # Enable offline mode for Hugging Face Hub
export TRANSFORMERS_OFFLINE=1 # Enable offline mode for Hugging Face Transformers

srun python test_instruct_qwen_infer.py