#!/usr/bin/env bash
set -e

# Baseline evaluation on the ORIGINAL Meta-Llama-3-8B-Instruct
# (no CTPT, no LoRA). Uses the SAME eval prompts and scoring
# as the CTPT+LoRA run, so the comparison is clean.

cd /mounts/work/yuetian_lu/hallulrh

export PYTHONPATH=$PWD/src:$PYTHONPATH

python -m hallulrh.eval.run_eval_baseline \
  --prompts experiments/sanity_check_1/eval/eval_prompts.jsonl \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --out-json experiments/sanity_check_1/eval/eval_chat_baseline.jsonl \
  --metrics-csv experiments/sanity_check_1/eval/metrics_baseline.csv
