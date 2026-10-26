#!/bin/bash
#SBATCH --job-name=kda_smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:29:00
#SBATCH --output=test_kda_%j.out
#SBATCH --error=test_kda_%j.err

source ~/distill_env_cu126/bin/activate

export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

python test_kda.py