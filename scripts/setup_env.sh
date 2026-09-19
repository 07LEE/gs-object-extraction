#!/usr/bin/env bash
# Install the application in an isolated Python environment.
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

VENV=${VENV:-.venv-gpu}
PYTHON=${PYTHON:-}
if [ -z "$PYTHON" ]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON=$candidate
      break
    fi
  done
fi
"${PYTHON:-python3}" -I -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 12)))' \
  || { echo 'Install Python 3.10–3.12 with venv support, or set PYTHON to its executable.' >&2; exit 1; }
for tool in git curl c++; do
  command -v "$tool" >/dev/null 2>&1 || { echo "Missing build tool: $tool" >&2; exit 1; }
done
NVCC=${CUDA_HOME:+$CUDA_HOME/bin/nvcc}
NVCC=${NVCC:-nvcc}
command -v "$NVCC" >/dev/null 2>&1 \
  || { echo 'Install the CUDA Toolkit (nvcc), or set CUDA_HOME to its directory.' >&2; exit 1; }
CUDA_VERSION=$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
case "$CUDA_VERSION" in
  11.8) TORCH=2.6.0; TORCHVISION=0.21.0; CUDA_WHEEL=cu118 ;;
  12.1) TORCH=2.5.1; TORCHVISION=0.20.1; CUDA_WHEEL=cu121 ;;
  12.4) TORCH=2.6.0; TORCHVISION=0.21.0; CUDA_WHEEL=cu124 ;;
  12.6) TORCH=2.7.1; TORCHVISION=0.22.1; CUDA_WHEEL=cu126 ;;
  12.8) TORCH=2.7.1; TORCHVISION=0.22.1; CUDA_WHEEL=cu128 ;;
  *) echo "Unsupported CUDA Toolkit: $CUDA_VERSION. Use 11.8, 12.1, 12.4, 12.6 or 12.8." >&2; exit 1 ;;
esac

# Preserve environments created by the old installer; never inherit their packages.
if [ -e "$VENV" ] || [ -L "$VENV" ]; then
  if ! [ -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -I -c '
import pathlib, sys
cfg = pathlib.Path(sys.prefix, "pyvenv.cfg").read_text().lower()
assert sys.prefix != sys.base_prefix
assert "include-system-site-packages = false" in cfg
assert (3, 10) <= sys.version_info[:2] <= (3, 12)
'; then
    backup=$(mktemp -d "${VENV}.backup.XXXXXX")
    mv -- "$VENV" "$backup/environment"
    echo "Previous environment saved to $backup/environment"
  fi
fi
[ -x "$VENV/bin/python" ] || "$PYTHON" -I -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip 'setuptools>=68' wheel ninja
"$PY" -m pip install "torch==$TORCH" "torchvision==$TORCHVISION" \
  --index-url "https://download.pytorch.org/whl/$CUDA_WHEEL"
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable; check the NVIDIA driver."'
"$PY" -m pip install --no-build-isolation --no-deps \
  'git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d'
# Keep later dependency resolution from replacing the CUDA build of PyTorch.
constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
printf 'torch==%s\ntorchvision==%s\n' "$TORCH" "$TORCHVISION" > "$constraints"
SAM2_BUILD_CUDA=0 "$PY" -m pip install --no-build-isolation -c "$constraints" -r requirements.txt
"$PY" -m pip install --no-build-isolation --no-deps -e .

CHECKPOINT_URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
CHECKPOINT=checkpoints/sam2.1_hiera_base_plus.pt
mkdir -p checkpoints
if [ ! -s "$CHECKPOINT" ]; then
  curl -L --fail --progress-bar -o "$CHECKPOINT.part" "$CHECKPOINT_URL"
  mv "$CHECKPOINT.part" "$CHECKPOINT"
fi
"$PY" -c "import torch, torchvision, PySide6, sam2, diff_gaussian_rasterization; print('Setup complete | torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
