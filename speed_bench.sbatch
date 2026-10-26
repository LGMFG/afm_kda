#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J speed_bench
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/afm_kda/logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/afm_kda/logs/%x-%j.err
 
# Inference speed benchmark (speed_bench.py).  The WHOLE node, one GPU used: no other job shares
# the node's CPUs, memory bandwidth or PCIe -- batch-1 decode is launch-bound and sensitive to that.
#   smoke (first):  sbatch --time=00:30:00 --export=ALL,SMOKE=1 speed_bench.sbatch
#   full:           sbatch speed_bench.sbatch           (resumable: resubmit the same line)
#   subset:         sbatch --export=ALL,MODELS=teacher:r1_kda_hybrid speed_bench.sbatch   (':' -- --export splits on ',')
source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate
export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS
export HF_HOME=$WORK_LUKAS/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
cd $WORK_LUKAS/afm_kda
 
echo "=== node $(hostname)   job $SLURM_JOB_ID   SMOKE=${SMOKE:-0}   MODELS=${MODELS:-all}"
nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm,power.limit --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv     # must be empty: nobody else on the GPU
python -c "import torch, transformers; from importlib.metadata import version; \
print('VERSIONS torch', torch.__version__, '| transformers', transformers.__version__, \
'| fla', version('flash-linear-attention'), '| flash_attn', version('flash_attn'))"
 
python speed_bench.py ${SMOKE:+--smoke}
echo "speed_bench exited $?"
