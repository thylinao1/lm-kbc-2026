#!/bin/bash
#SBATCH --job-name=akbc-cap
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/e/e1506804/akbc/logs/cap_%j.out
#SBATCH --error=/home/e/e1506804/akbc/logs/cap_%j.err
set -uo pipefail
echo "node=$(hostname) start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export AKBC_DATA=$HOME/akbc/dataset2026/data
export AKBC_POOLS=$HOME/akbc/pools
export PATH=$HOME/miniconda3/envs/csa-vllm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=WARNING
PY=$HOME/miniconda3/envs/csa-vllm/bin/python
cd $HOME/akbc
REV=5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89

# One model load for all 18 pools. hasCapacity is the campaign: the field best
# on TEST is 0.3265 and the #1 published pool coverage ~0.77 against realised
# ~0.38, having exhausted 24+ AGGREGATION mechanisms. These five frames attack
# the GENERATION side; consensus.py then tests cross-frame agreement, which is
# a signal no single-frame system can have.
$PY pipeline/generate_many.py \
    --channels cap_infobox,cap_disambig,cap_current,cap_listing,cap_recite,death_obituary,death_direct,stock_sentinel,area_lead \
    --splits train,val --n 30 --revision $REV --max-model-len 8192 --gpu-mem-util 0.90
echo "GENERATE_EXIT=$?"

echo "########## coverage: selection-bound or knowledge-bound? ##########"
for CH in cap_infobox cap_disambig cap_current cap_listing cap_recite; do
  $PY pipeline/coverage.py --channel $CH --split val --param 1.0 --revision $REV || echo "COVFAIL $CH"
done

echo "########## CROSS-FRAME CONSENSUS (the experiment) ##########"
$PY pipeline/consensus.py --relation hasCapacity \
    --channels cap_infobox,cap_disambig,cap_current,cap_listing,cap_recite \
    --splits train,val --revision $REV || echo "CONSFAIL"

echo "########## per-channel tuning ##########"
for CH in cap_infobox cap_disambig cap_current cap_listing cap_recite death_obituary death_direct stock_sentinel area_lead; do
  $PY pipeline/tune.py --channel $CH --revision $REV || echo "TUNEFAIL $CH"
done
echo "ALLDONE=$(date)"
