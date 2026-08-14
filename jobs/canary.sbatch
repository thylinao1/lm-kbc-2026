#!/bin/bash
#SBATCH --job-name=akbc-canary
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/e/e1506804/akbc/logs/canary_%j.out
#SBATCH --error=/home/e/e1506804/akbc/logs/canary_%j.err
set -euo pipefail
echo "node=$(hostname) start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Closed-book: weights are already staged; no network reads at inference time.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export AKBC_DATA=$HOME/akbc/dataset2026/data
export AKBC_POOLS=$HOME/akbc/pools
export PATH=$HOME/miniconda3/envs/csa-vllm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=WARNING
PY=$HOME/miniconda3/envs/csa-vllm/bin/python
cd $HOME/akbc

REV=5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89

for SPLIT in train val; do
  echo "=== generating borders_list / $SPLIT ==="
  $PY pipeline/generate.py --channel borders_list --split $SPLIT --n 30 \
      --revision $REV --max-model-len 4096 --gpu-mem-util 0.90
done

echo "=== tuning tau on train, reporting val ==="
$PY pipeline/tune.py --channel borders_list --revision $REV --python $PY

echo "done=$(date)"
