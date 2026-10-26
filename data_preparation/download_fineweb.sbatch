#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -p lrd_all_serial
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:59:00
#SBATCH -J fineweb_dl
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/download_fineweb.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/download_fineweb.err

export HF_HOME=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache
source ~/distill_env_cu126/bin/activate
cd /leonardo_work/EUHPC_D34_049/WORK_LUKAS

python download_fineweb_tokens_lowmem.py \
  --target-tokens 3e9 \
  --tokenizer arcee-ai/AFM-4.5B-Base \
  --out-dir /leonardo_work/EUHPC_D34_049/WORK_LUKAS/datasets/fineweb_edu_raw \
  --seed 42