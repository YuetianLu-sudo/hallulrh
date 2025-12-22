
This document describes, end-to-end, how we produced the figures:
1) **Behavior rates by relation** (hallucination vs refusal; with uncertainty bars)
2) **Linearity proxy vs hallucination** (scatter of Δcos vs hallucination rate; pooled across model×relation)

---

## 0. What the figures show

### Figure 1: Behavior rates by relation (per model)
For each model and relation/task, we compute:
- hallucination rate
- refusal rate

The plot displays these rates (typically as stacked horizontal bars), with 95% Wilson confidence intervals.

### Figure 2: LRE linearity proxy vs hallucination (pooled scatter)
For each (model, relation) pair we compute:
- x-axis: **Δcos = LRE cosine improvement** (difference-vector LRE proxy for linearity)
- y-axis: **hallucination rate** (from judge labels)

We also report Spearman correlation across all points (pooled).

---

## 1. Inputs (files and expected schema)

### 1.1 LRE prompt file (subject–relation–object triples)
**File**: `data/lre/natural_relations_ext6_q_fname.jsonl`

Each line is a JSON object with at least:
- `relation` : string (e.g., father, instrument, sport, company_ceo, company_hq, country_language)
- `text`     : prompt text (we use question-style prompts in our experiments)
- `subject`  : subject string span
- `answer`   : gold object string span (used for locating object tokens)

This JSONL is used only to extract representations from the LM (not to run the judge).

### 1.2 Judge outputs (behavior labels)
We use judge CSVs produced by Gemini where labels are constrained to **two classes**:
- `hallucination`
- `refusal`

**Typical files** (example paths; adjust to repo reality):
- `data/judge/future/<MODEL>_future_with_judge.csv`  
  contains tasks: `father`, `instrument`
- `data/judge/relpanel_q/<MODEL>_relpanel_with_judge.csv`  
  contains tasks: `company_ceo`, `company_hq`, `country_language`, `sport`

Required columns:
- `task` (or `relation`) : relation name
- `model_name`           : short model name used consistently everywhere
- `judge_label`          : raw label string (may be upper-case; we normalize)

Additional columns:
- `question`, `answer`, `judge_reason`, etc.

**Label normalization**:
We normalize judge labels via:
`judge_label_norm = str(judge_label).strip().lower()`

After normalization, we expect only two values:
- `hallucination`
- `refusal`

---

## 2. Step A — Compute LRE linearity proxy (Δcos) per model × relation

### 2.1 Script
**Script**: `scripts/run_lre_relpanel.py`  
(This is the difference-vector LRE implementation currently used in main results.)

### 2.2 Method summary (difference-vector LRE)
For a fixed model and a relation r, we extract n pairs of vectors:
- subject representations: s_i ∈ R^d
- object representations:  o_i ∈ R^d

We estimate a single relation direction vector d̄_r using a train split T:
- per-sample difference: d_i = o_i − s_i
- relation direction:   d̄_r = (1/|T|) Σ_{i∈T} (o_i − s_i)

For each test sample j in E:
- predicted object vector:  ô_j = s_j + d̄_r

We report:
- baseline cosine: cos_base = E_{j∈E}[ cos(s_j, o_j) ]
- LRE cosine:      cos_lre  = E_{j∈E}[ cos(ô_j, o_j) ]
- **Δcos**:         Δcos = cos_lre − cos_base
- MSE:             E_{j∈E}[ ||ô_j − o_j||_2^2 ]

This corresponds to an affine map with W = I and b = d̄_r:
ô = Ws + b.

### 2.3 Representation extraction details
For each example we build:
`full_text = text + " " + answer`

We tokenize with offset mapping and locate:
- subject token span: first occurrence of `subject` in `full_text`
- object token span:  last occurrence of `answer` in `full_text` (using rfind to avoid collisions)

We extract hidden states from two layers:
- `subject_layer = num_layers // 2`
- `object_layer  = max(subject_layer + 2, num_layers - 2)`

We mean-pool token vectors for the span to get s_i and o_i.

### 2.4 Repro command (per model)
Example (replace HF ids / devices as needed):

```bash
# Gemma
python scripts/run_lre_relpanel.py \
  --model-id <HF_ID_FOR_GEMMA_7B_IT> \
  --model-name gemma_7b_it \
  --prompts data/lre/natural_relations_ext6_q_fname.jsonl \
  --output-csv data/lre/natural_by_model_ext6_q_fname/gemma_7b_it.csv

# Llama 3.1
python scripts/run_lre_relpanel.py \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --model-name llama3_1_8b_instruct \
  --prompts data/lre/natural_relations_ext6_q_fname.jsonl \
  --output-csv data/lre/natural_by_model_ext6_q_fname/llama3_1_8b_instruct.csv

# Mistral
python scripts/run_lre_relpanel.py \
  --model-id <HF_ID_FOR_MISTRAL_7B_INSTRUCT> \
  --model-name mistral_7b_instruct \
  --prompts data/lre/natural_relations_ext6_q_fname.jsonl \
  --output-csv data/lre/natural_by_model_ext6_q_fname/mistral_7b_instruct.csv

# Qwen
python scripts/run_lre_relpanel.py \
  --model-id <HF_ID_FOR_QWEN2_5_7B_INSTRUCT> \
  --model-name qwen2_5_7b_instruct \
  --prompts data/lre/natural_relations_ext6_q_fname.jsonl \
  --output-csv data/lre/natural_by_model_ext6_q_fname/qwen2_5_7b_instruct.csv
```

### 2.5 Outputs
Each run writes a per-relation summary CSV with columns including:
- `model_name`, `relation`, `num_layers`, `subject_layer`, `object_layer`
- `n_total`, `n_train`, `n_test`
- `base_cos_mean`, `cos_mean`, `cos_improvement` (Δcos), `mse`

---

## 3. Step B — Compute behavior rates from judge CSVs

For each (model_name, task/relation) we compute:
- N = total judged samples
- hallucination_count, refusal_count
- hallucination_rate = hallucination_count / N
- refusal_rate = refusal_count / N
- 95% Wilson CI for each rate (used as error bars in Figure 1)
