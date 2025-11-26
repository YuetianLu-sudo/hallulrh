#!/usr/bin/env bash
set -euo pipefail

# This script creates the project directory layout for the hallulrh project.
# Run it from the project root directory.

########################
# Create directories
########################

mkdir -p env
mkdir -p configs

mkdir -p data/raw/corpus_a_women
mkdir -p data/raw/corpus_b_musicians
mkdir -p data/metadata
mkdir -p data/processed/tokenized
mkdir -p data/processed/eval_sets

mkdir -p src/hallulrh
mkdir -p src/hallulrh/data
mkdir -p src/hallulrh/models
mkdir -p src/hallulrh/eval
mkdir -p src/hallulrh/analysis

mkdir -p scripts

mkdir -p experiments/sanity_check_1/cpt_l3_8b_chat/lora_ckpt
mkdir -p experiments/sanity_check_1/cpt_l3_8b_base/lora_ckpt
mkdir -p experiments/sanity_check_1/eval/plots

########################
# Top-level files
########################

touch README.md
touch pyproject.toml
touch .gitignore
touch .env.example

########################
# Env spec files
########################

touch env/environment.yml
touch env/requirements.txt

########################
# Config files
########################

touch configs/cpt_l3_8b_instruct.yaml
touch configs/eval_l3_8b_instruct.yaml
touch configs/probes.yaml

########################
# Data metadata files
########################

touch data/metadata/entities.csv
touch data/metadata/anchor_prompts.json
touch data/metadata/refusal_probes.json

########################
# Src package files
########################

# Top-level package
touch src/hallulrh/__init__.py
touch src/hallulrh/config.py
touch src/hallulrh/logging_utils.py
touch src/hallulrh/seed_utils.py

# Data subpackage
touch src/hallulrh/data/__init__.py
touch src/hallulrh/data/distinctly__bios_generate.py
touch src/hallulrh/data/datasets.py
touch src/hallulrh/data/tokenization.py

# Models subpackage
touch src/hallulrh/models/__init__.py
touch src/hallulrh/models/model_loader.py
touch src/hallulrh/models/lora_setup.py
touch src/hallulrh/models/cpt_trainer.py
touch src/hallulrh/models/probes.py

# Eval subpackage
touch src/hallulrh/eval/__init__.py
touch src/hallulrh/eval/prompts.py
touch src/hallulrh/eval/decoding.py
touch src/hallulrh/eval/scoring.py
touch src/hallulrh/eval/metrics.py

# Analysis subpackage
touch src/hallulrh/analysis/__init__.py
touch src/hallulrh/analysis/plot_results.py
touch src/hallulrh/analysis/summary_tables.py

########################
# Scripts
########################

touch scripts/run_cpt_sanity.sh
touch scripts/run_eval_sanity.sh
touch scripts/inspect_outputs.sh

chmod +x scripts/run_cpt_sanity.sh
chmod +x scripts/run_eval_sanity.sh
chmod +x scripts/inspect_outputs.sh

########################
# Experiments files
########################

# Chat CTPT run
touch experiments/sanity_check_1/cpt_l3_8b_chat/config.yaml
touch experiments/sanity_check_1/cpt_l3_8b_chat/train.log
touch experiments/sanity_check_1/cpt_l3_8b_chat/metrics.json

# Base CTPT run (control)
touch experiments/sanity_check_1/cpt_l3_8b_base/config.yaml
touch experiments/sanity_check_1/cpt_l3_8b_base/train.log
touch experiments/sanity_check_1/cpt_l3_8b_base/metrics.json

# Eval outputs
touch experiments/sanity_check_1/eval/eval_chat.jsonl
touch experiments/sanity_check_1/eval/eval_base.jsonl
touch experiments/sanity_check_1/eval/metrics.csv
