#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J ce_outliers
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/afm_kda/logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/afm_kda/logs/%x-%j.err

# Why does R5 look +0.31 nats worse in test_r1_ce_math.py but level with R1
# in DistillKit's own eval?  Per-sequence, per-position, sdpa vs FA2, per-layer
# magnitudes, and weight drift -- one job.
#
#   sbatch --export=ALL,TULU3=/path/to/tulu3 ce_outliers.sbatch
#
# TULU3 is optional (web alone answers the main question).  Use the same path
# as the test_r1_ce_math.py run.  It is read, never assigned, here.

source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate
export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS
export HF_HOME=$WORK_LUKAS/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TULU3=${TULU3:-}

cd $WORK_LUKAS/afm_kda
echo "TULU3=${TULU3:-<not set: web only>}"
python ce_outliers.py
echo "ce_outliers exited $?"