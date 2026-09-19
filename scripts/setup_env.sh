#!/usr/bin/env bash
# Build a virtual environment on top of the Graphdeco 3DGS training environment and install
# requirements.txt, this package (editable) and the SAM2 checkpoint.
#
#   GS_PYTHON=/path/to/gs_train/bin/python VENV=.venv-gpu scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH  # paths from other Python installs (e.g. ROS) break the environment

GS_PYTHON=${GS_PYTHON:-}
VENV=${VENV:-.venv-gpu}

if [ -z "$GS_PYTHON" ]; then
  echo "set GS_PYTHON to the python of the environment that trains Graphdeco 3DGS, the one" >&2
  echo "where 'import torch, torchvision, diff_gaussian_rasterization' works, for example" >&2
  echo "  GS_PYTHON=~/miniconda3/envs/gs_train/bin/python scripts/setup_env.sh" >&2
  exit 1
fi
[ -x "$GS_PYTHON" ] || { echo "GS_PYTHON is not an executable: $GS_PYTHON" >&2; exit 1; }
CHECKPOINT_URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
CHECKPOINT=checkpoints/sam2.1_hiera_base_plus.pt

[ -x "$VENV/bin/python" ] || "$GS_PYTHON" -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"

python -c "import torch, torchvision, diff_gaussian_rasterization" \
  || { echo "base environment lacks torch, torchvision or diff_gaussian_rasterization" >&2; exit 1; }
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .

mkdir -p checkpoints
if [ ! -f "$CHECKPOINT" ]; then
  curl -L --fail --progress-bar -o "$CHECKPOINT.part" "$CHECKPOINT_URL"
  mv "$CHECKPOINT.part" "$CHECKPOINT"
fi

python -c "import torch, PySide6, sam2, diff_gaussian_rasterization; print('torch', torch.__version__, '| PySide6', PySide6.__version__, '| cuda', torch.cuda.is_available())"
