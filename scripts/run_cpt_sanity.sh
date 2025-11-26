#!/usr/bin/env bash
set -e

# Run minimal CTPT (LM + LoRA) sanity check on Llama-3-8B-Instruct.
# Assumes the 'hallulrh' conda env is already activated in the shell.

cd /mounts/work/yuetian_lu/hallulrh

export PYTHONPATH=$PWD/src:$PYTHONPATH

python -m hallulrh.models.cpt_trainer \
  --config configs/cpt_l3_8b_instruct.yaml \
  --experiment-dir experiments/sanity_check_1/cpt_l3_8b_chat
