#!/bin/bash
#SBATCH --job-name=akbc-test
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/e/e1506804/akbc/logs/test_%j.out
#SBATCH --error=/home/e/e1506804/akbc/logs/test_%j.err
set -uo pipefail
echo "node=$(hostname) start=$(date)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export AKBC_DATA=$HOME/akbc/dataset2026/data
export AKBC_POOLS=$HOME/akbc/pools
export PATH=$HOME/miniconda3/envs/csa-vllm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=WARNING
PY=$HOME/miniconda3/envs/csa-vllm/bin/python
cd $HOME/akbc
REV=5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89

# TEST pools. Nothing can be submitted without these; only train and val existed.
# Generated for every channel that is a live candidate, so the final config can
# be chosen offline from cached pools without another GPU round.
$PY pipeline/generate_many.py \
    --channels borders_list,death_obituary,death_direct,stock_sentinel,area_lead,area_listing,area_infobox,award_list \
    --splits test --n 30 --revision $REV --max-model-len 8192 --gpu-mem-util 0.90

# capacity gets the deeper pool, matching train/val
$PY pipeline/generate_many.py --channels cap_recite,cap_current,cap_disambig \
    --splits test --n 100 --revision $REV --max-model-len 8192 --gpu-mem-util 0.90
echo "ALLDONE=$(date)"
