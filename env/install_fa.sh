#!/usr/bin/env bash
set -euo pipefail

# ---- environment ----
source ~/distill_env_cu126/bin/activate

module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.6

# Paths
WORK_LUKAS="${WORK_LUKAS:-/leonardo_work/EUHPC_D34_049/WORK_LUKAS}"
WHEELHOUSE="$WORK_LUKAS/wheelhouse/flashattn_cu126"
SRC_DIR="$WORK_LUKAS/src/flash-attention-2.8.3"

mkdir -p "$WHEELHOUSE" "$WORK_LUKAS/src"

# Clone flash-attention source if missing
if [ ! -d "$SRC_DIR/.git" ]; then
  git clone --recursive --branch v2.8.3 \
    https://github.com/Dao-AILab/flash-attention.git \
    "$SRC_DIR"
else
  git -C "$SRC_DIR" fetch --tags
  git -C "$SRC_DIR" checkout v2.8.3
  git -C "$SRC_DIR" submodule update --init --recursive
fi

# Download Python wheel dependencies
python -m pip download -d "$WHEELHOUSE" \
  ninja psutil packaging einops setuptools wheel
