#!/usr/bin/env bash
set -e

# Sanity-check eval:
# 1) build eval prompts from entities.csv
# 2) run CTPT+LoRA eval on Llama-3-8B-Instruct

cd /mounts/work/yuetian_lu/hallulrh

# Ensure Python can see the hallulrh package
export PYTHONPATH=$PWD/src:$PYTHONPATH

echo "[hallulrh] Step 1: build eval prompts"
python src/hallulrh/eval/prompts.py

echo "[hallulrh] Step 2: CTPT+LoRA eval (Llama-3-8B-Instruct + LoRA step_50)"
python -m hallulrh.eval.run_eval \
  --prompts experiments/sanity_check_1/eval/eval_prompts.jsonl \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --lora-ckpt experiments/sanity_check_1/cpt_l3_8b_chat/lora_ckpt/step_50 \
  --out-json experiments/sanity_check_1/eval/eval_chat.jsonl \
  --metrics-csv experiments/sanity_check_1/eval/metrics.csv


