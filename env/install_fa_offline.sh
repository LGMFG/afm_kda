set -euo pipefail # Exit immediately if a command exits with a non-zero status, if an undefined variable is used, or if any command in a pipeline fails.

module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.6

source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate

WORK_LUKAS="${WORK_LUKAS:-/leonardo_work/EUHPC_D34_049/WORK_LUKAS}"
WHEELHOUSE="$WORK_LUKAS/wheelhouse/flashattn_cu126"
SRC_DIR="$WORK_LUKAS/src/flash-attention-2.8.3"

mkdir -p "$WHEELHOUSE/tmp" "$WHEELHOUSE/cache"

export TMPDIR="$WHEELHOUSE/tmp"
export PIP_CACHE_DIR="$WHEELHOUSE/cache"

export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export CC="$(which gcc)"
export CXX="$(which g++)"

export TORCH_CUDA_ARCH_LIST="8.0"
export FLASH_ATTN_CUDA_ARCHS="80"

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE

export MAX_JOBS=8 # parallel build jobs. Tune this based on your system's CPU cores and memory.

python -m pip uninstall -y flash-attn

python -m pip install --no-index --find-links="$WHEELHOUSE" \
  ninja psutil packaging einops setuptools wheel

# build flash-attn from local source
git -C "$SRC_DIR" submodule update --init --recursive

python -m pip install --no-build-isolation --no-deps "$SRC_DIR"

echo "FLASH ATTENTION INSTALLED"