#!/bin/bash
#SBATCH --job-name=akbc-dl
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/e/e1506804/akbc/logs/dl_%j.out
#SBATCH --error=/home/e/e1506804/akbc/logs/dl_%j.err
set -euo pipefail
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
echo "node=$(hostname) start=$(date)"
df -h /home | tail -1
$HOME/miniconda3/envs/csa-vllm/bin/python - << "PY"
from huggingface_hub import snapshot_download
p = snapshot_download("google/gemma-4-31B",
    allow_patterns=["*.safetensors","*.json","*.model","*.txt"],
    max_workers=4)
print("SNAPSHOT", p)
PY
echo "done=$(date)"
du -sh $HOME/.cache/huggingface/hub/models--google--gemma-4-31B
