#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J heldout_kl
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/afm_kda/logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/afm_kda/logs/%x-%j.err

# Held-out CE + KL vs the teacher on FineWeb-Edu AND FineMath, every model, one table.
#   sbatch heldout_kl.sbatch                                   default list (resubmit until "0 models to measure")
#   sbatch --export=ALL,MODELS_FILE=/path/list.txt heldout_kl.sbatch   any name=path list (checkpoint curves)
# Models whose JSON exists in eval_results/heldout_kl/ are skipped, so a job killed at 30 min resumes.
source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate
export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS
export HF_HOME=$WORK_LUKAS/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
cd $WORK_LUKAS/afm_kda
python heldout_kl.py
echo "heldout_kl exited $?"