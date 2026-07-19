#!/bin/bash
# Build the vllm venv for the cross-engine case study.
# Paper environment: vllm 0.19.1, torch 2.10.0+cu128, python 3.11, NVIDIA GH200 (aarch64).
# The module line below is TACC Vista specific — replace with your site's toolchain
# (any CUDA 12.8 + python 3.11 environment).
set -e

# . /opt/apps/lmod/lmod/init/profile && module reset && module load gcc/14.2.0 cuda/12.8 python3/3.11.8   # TACC Vista

ENV_ROOT=${ENV_ROOT:-$HOME/envs}
mkdir -p $ENV_ROOT
cd $ENV_ROOT

if [ ! -d llm-vllm ]; then
  echo "[$(date +%H:%M:%S)] creating venv llm-vllm"
  python3 -m venv llm-vllm
fi
source llm-vllm/bin/activate
echo "[$(date +%H:%M:%S)] python: $(which python)  $(python --version)"

echo "[$(date +%H:%M:%S)] pip upgrade"
pip install --quiet --upgrade pip wheel setuptools

echo "[$(date +%H:%M:%S)] install vllm (paper version)"
pip install "vllm==0.19.1" 2>&1 | tail -30

echo "[$(date +%H:%M:%S)] smoke test"
python -c "import vllm; print('vllm version:', vllm.__version__)" 2>&1
echo "[$(date +%H:%M:%S)] DONE  (if torch/nvidia-lib versions conflict, run fix_torch_v4.sh $ENV_ROOT/llm-vllm)"
