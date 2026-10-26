#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J get_finemath
#SBATCH -p lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/%x-%j.err

# FineMath-4plus -> three splits in $WORK_LUKAS/datasets, like FineWeb-Edu.
# Budget-free serial partition (4 cores, 15.4 GB RAM, 4 h, internet).  See claude/download_finemath.md.
#   sbatch --export=ALL,STEP=download get_finemath.sbatch
#   sbatch --export=ALL,STEP=prepare  get_finemath.sbatch      (after the download log says "manifest ->")
# Optional: MARGIN=1.6 (download only: more docs), S1_TOKENS / S2_TOKENS (defaults 3e8 / 1e9).
: "${STEP:?submit with --export=ALL,STEP=download|prepare}"

source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate
export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS
export HF_HOME=$WORK_LUKAS/hf_cache
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE   # download needs the hub
export TOKENIZERS_PARALLELISM=false

# tokenizer from the local teacher snapshot: no hub call for the gated AFM repo
TOK=$WORK_LUKAS/hf_cache/hub/models--arcee-ai--AFM-4.5B-Base/snapshots/dab7922f9c868d479b365410304f466b007b1c5a
[ -f $TOK/tokenizer.json ] || [ -f $TOK/tokenizer_config.json ] || { echo "FATAL: no tokenizer in $TOK"; exit 1; }

D=$WORK_LUKAS/datasets
RAW=$D/finemath_4plus_raw            # like fineweb_edu_raw
OUT=$D/finemath_4plus                # -> _heldout, _stage1, _stage2, _stats.json
CACHE=$D/_arrow_tmp_finemath         # intermediate Arrow cache, delete after prepare
TOKS="--stage1-tokens ${S1_TOKENS:-3e8} --stage2-tokens ${S2_TOKENS:-1e9} --tokenizer $TOK"
case "$STEP" in
  download) python get_finemath.py download --out $RAW --margin ${MARGIN:-1.35} $TOKS ;;
  prepare)  python get_finemath.py prepare --raw $RAW --out $OUT --num-proc 4 --cache-dir $CACHE $TOKS ;;
  *) echo "FATAL: STEP=$STEP"; exit 1 ;;
esac
RC=$?
echo "exit $RC"
du -sh $RAW $CACHE ${OUT}_heldout ${OUT}_stage1 ${OUT}_stage2 2>/dev/null
exit $RC