Sanity-check experiments for "Do Linearizable Properties Encourage Hallucinations?"

- sanity_check_1/
    Current / canonical version of the father vs. instrument sanity check.
    - Uses the latest v3 synthetic corpus (no explicit "no information" sentences).
    - Contains CPT (LoRA) runs and eval JSONL for Llama-3-8B-Instruct:
        * cpt_l3_8b_base/   (optional base CPT)
        * cpt_l3_8b_chat/   (main CTPT+LoRA run)
        * eval/
            - eval_base.jsonl
            - eval_chat_baseline.jsonl   # Llama-3-8B-Instruct, no CTPT
            - eval_chat.jsonl            # Llama-3-8B-Instruct + CTPT+LoRA
            - metrics_baseline.csv
            - metrics.csv
            - plots/
    → All numbers in the pilot write-up and ACL outline currently refer to this directory
      (plus its LM-as-judge summaries in data/).

- sanity_check_1_v1_disclaimer/
    Older pilot version (historical, not used in the paper).
    - Early synthetic corpus with explicit "no information" / disclaimer sentences
      in the biographies.
    - Kept only so that the research trajectory is reproducible.

- sanity_check_1_v2_l3_8b_instruct/
    Intermediate pilot (historical, not used in the paper).
    - First run where Llama-3-8B-Instruct was evaluated with the newer corpus,
      before the current setup in sanity_check_1/ was finalized.
    - Directory structure mirrors sanity_check_1/, but results are superseded
      by the ones in sanity_check_1/.

General convention:
- New sanity checks for other relations (e.g. sport, language, company→CEO)
  should either:
  * live under this experiment as new eval files
    (e.g. eval_l3_8b_instruct_sport.jsonl), or
  * get their own subdirectory, e.g. `sanity_check_2_company_ceo/`,
  while keeping the same internal layout: cpt_*/ + eval/.
