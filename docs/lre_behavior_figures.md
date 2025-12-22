
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
