import pandas as pd
from pathlib import Path

p = Path("data/judge/future/qwen2_5_7b_instruct_baseline_with_judge.csv")

df = pd.read_csv(p)

sub = df[df["task"] == "father"].head(5)

print("=== sanity check: saved question/answer samples ===")
for i, row in sub.iterrows():
    q = str(row["question"])
    a = str(row["answer"])
    print("\n---")
    print("Q:", q)
    print("A_prefix:", a[:120].replace("\n","\\n"))
