#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -p lrd_all_serial
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -J fineweb_prep
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/prepare_fineweb.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/prepare_fineweb.err

export HF_HOME=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache
source ~/distill_env_cu126/bin/activate
cd /leonardo_work/EUHPC_D34_049/WORK_LUKAS

python prepare_fineweb_dataset.py \
  --raw-dir  /leonardo_work/EUHPC_D34_049/WORK_LUKAS/datasets/fineweb_edu_raw \
  --out-dir  /leonardo_work/EUHPC_D34_049/WORK_LUKAS/datasets/fineweb_edu_3B \
  --cache-dir /leonardo_work/EUHPC_D34_049/WORK_LUKAS/datasets/_arrow_tmp \
  --num-proc 4