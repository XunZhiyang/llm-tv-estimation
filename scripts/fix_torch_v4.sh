#!/bin/bash
# Pin the nvidia-* wheel versions to exactly what torch 2.10.0+cu128 declares.
# Needed on aarch64 (GH200) where engine installs can pull newer, incompatible
# nvidia libs. Usage:  bash scripts/fix_torch_v4.sh <path-to-venv>
# The module line below is TACC Vista specific — replace with your site's toolchain.

# . /opt/apps/lmod/lmod/init/profile && module reset && module load gcc/14.2.0 cuda/12.8 python3/3.11.8   # TACC Vista
source $1/bin/activate

# torch 2.10.0+cu128 pins:
echo "[$(date +%H:%M:%S)] downgrade nvidia libs to torch's exact requirements"
pip install --quiet \
    'nvidia-cusparselt-cu12==0.7.1' \
    'nvidia-nccl-cu12==2.27.5' \
    'nvidia-nvjitlink-cu12==12.8.93' \
    'nvidia-nvshmem-cu12==3.4.5' \
    'nvidia-nvtx-cu12==12.8.90' \
    'nvidia-cublas-cu12==12.8.4.1' \
    'nvidia-cuda-cupti-cu12==12.8.90' \
    'nvidia-cuda-nvrtc-cu12==12.8.93' \
    'nvidia-cuda-runtime-cu12==12.8.90' \
    'nvidia-cudnn-cu12==9.10.2.21' \
    'nvidia-cufft-cu12==11.3.3.83' \
    'nvidia-curand-cu12==10.3.9.90' \
    'nvidia-cusolver-cu12==11.7.3.90' \
    'nvidia-cusparse-cu12==12.5.8.93' \
    'nvidia-cufile-cu12==1.13.1.3' \
    'triton==3.5.0' 2>&1 | tail -5

echo "[$(date +%H:%M:%S)] verify torch"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'cuda v', torch.version.cuda)" 2>&1 | tail -5
echo "[$(date +%H:%M:%S)] verify engine import"
case "$1" in
    *vllm*) python -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -5 ;;
    *sglang*) python -c "import sglang; print('sglang', sglang.__version__)" 2>&1 | tail -5 ;;
esac
echo "[$(date +%H:%M:%S)] done"
