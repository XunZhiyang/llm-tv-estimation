#!/bin/bash
#SBATCH -J tv-engines
#SBATCH -p gh                      # single GH200 node (TACC Vista); adapt to your cluster
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A YOUR_ALLOCATION
#SBATCH -o logs/tv-engines-%j.out  # logs/ must exist at submit time (kept in the repo via logs/.gitkeep)

# Cross-engine TV production run: replay + seq oracles (paper reference campaign).
# Set PROJECT to the repo checkout and the two venv paths before submitting.
export PROJECT=${PROJECT:-$SLURM_SUBMIT_DIR}
export VLLM_VENV=${VLLM_VENV:?set VLLM_VENV}
export SGLANG_VENV=${SGLANG_VENV:?set SGLANG_VENV}
bash $PROJECT/scripts/run_decode_production.sh
