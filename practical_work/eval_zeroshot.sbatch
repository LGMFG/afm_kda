#!/bin/bash
#SBATCH --account=EUHPC_D34_049
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:29:59
#SBATCH -p boost_usr_prod
#SBATCH -J tulu3_full_4B_instr_06B_base_sl4096_bs2_gacc2_lr1e-4_mgn3_ce03_kl07_t2_4gpu_zeroshot_nochat
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

export PYTHONDONTWRITEBYTECODE=1 # my attempt to avoid writing .pyc files to the shared filesystem, which can be slow and cause issues


echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

ls -lah $HF_DATASETS_CACHE | head
find $HF_DATASETS_CACHE -maxdepth 2 | grep ai2_arc


echo "Starting at $(date)"
nvidia-smi


# Run lm-eval-harness with the specified config file
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_4B_instr_06B/checkpoint-21077,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_10k_4B_instr_06B/checkpoint-226,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_4B_instr_06B_sl2048_bs2_gacc4/checkpoint-2000,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_4B_instr_06B_sl2048_bs2_gacc4_lr3e-5_mgn1_ce03_kl07_hsc00/checkpoint-2000,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_4B_instr_1.7B_sl2048_bs2_gacc4_lr3e-5_mgn1_ce03_kl03_hsc04/checkpoint-2000,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_4B_instr_4B_Base_sl2048_bs2_gacc4_lr3e-5_mgn1_ce03_kl03_hsc04/checkpoint-600,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/dolci_4B_instr_06B_sl1024_bs2_gacc4_lr2e-4_mgn3_ce02_kl04_hsc04_t2/checkpoint-2000,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/hf_cache/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_4B_instr_06B_sl2048_bs2_gacc4_lr2e-4_mgn3_ce02_kl04_hsc04_t2/checkpoint-2000,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_4B_instr_06B_base_sl4096_bs1_gacc4_lr1e-4_mgn3_ce03_kl05_hsc02_t2/checkpoint-2000,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/dolci_4B_instr_06B_base_sl4096_bs1_gacc4_lr1e-4_mgn3_ce03_kl05_hsc02_t2/checkpoint-2000,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/dolci_4B_instr_06B_base_sl4096_bs1_gacc4_lr1e-4_mgn5_ce09_kl01_hsc00_t2/checkpoint-2000,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_4B_instr_06B_base_sl4096_bs1_gacc4_lr1e-4_mgn5_ce09_kl01_hsc00_t2/checkpoint-2000,dtype=bfloat16 \

#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/dolci_full_4B_instr_06B_base_sl4096_bs2_gacc2_lr1e-4_mgn3_ce03_kl07_t2_4gpu/checkpoint-19348,dtype=bfloat16 \

#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_full_4B_instr_06B_base_sl4096_bs2_gacc2_lr1e-4_mgn3_ce07_kl03_t2_4gpu/checkpoint-10075,dtype=bfloat16 \
#--model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_full_4B_instr_06B_base_sl4096_bs2_gacc2_lr1e-4_mgn3_ce03_kl07_t2_4gpu/checkpoint-10075,dtype=bfloat16 \

#--apply_chat_template \
srun lm_eval \
  --model hf \
  --model_args pretrained=/leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/output/tulu3_full_4B_instr_06B_base_sl4096_bs2_gacc2_lr1e-4_mgn3_ce03_kl07_t2_4gpu/checkpoint-10075,dtype=bfloat16 \
  --tasks arc_challenge,hellaswag,winogrande \
  --batch_size auto \
  --num_fewshot 0 \
  --output_path /leonardo_work/EUHPC_D34_049/WORK_LUKAS/eval_results/tulu3_full_4B_instr_06B_base_sl4096_bs2_gacc2_lr1e-4_mgn3_ce03_kl07_t2_4gpu_zeroshot_nochat

echo "Finished at $(date)"