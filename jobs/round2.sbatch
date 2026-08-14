#!/bin/bash
#SBATCH --job-name=akbc-r2
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/e/e1506804/akbc/logs/r2_%j.out
#SBATCH --error=/home/e/e1506804/akbc/logs/r2_%j.err
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

# Round 2 priorities, chosen from measured gaps rather than from the plan:
#  hasArea is our largest shortfall vs the field (0.76 vs a published 0.85) and
#  had only ONE frame; award had NO channel at all and is worth +0.0078 free.
$PY pipeline/generate_many.py --channels area_listing,area_infobox,award_list \
    --splits train,val --n 30 --revision $REV --max-model-len 8192 --gpu-mem-util 0.90

# Deepen the two best capacity frames to n=100. Draws are append-only and n is
# not part of POOL-ID, so this adds 70 per subject rather than regenerating.
# Coverage at n=30 is already 0.804 on cap_recite; more draws should raise both
# coverage and the concentration that predicts correctness.
$PY pipeline/generate_many.py --channels cap_recite,cap_current \
    --splits train,val --n 100 --revision $REV --max-model-len 8192 --gpu-mem-util 0.90

for CH in area_listing area_infobox award_list cap_recite cap_current; do
  $PY pipeline/tune.py --channel $CH --revision $REV || echo "TUNEFAIL $CH"
done
for CH in cap_recite cap_current; do
  $PY pipeline/coverage.py --channel $CH --split val --param 1.0 --revision $REV || echo "COVFAIL $CH"
done
$PY pipeline/consensus.py --relation hasArea --channels area_lead,area_listing,area_infobox \
    --splits train,val --revision $REV || echo "CONSFAIL area"
echo "ALLDONE=$(date)"
