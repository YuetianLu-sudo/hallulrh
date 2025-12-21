import pandas as pd
from pathlib import Path


INPUT_PATH = "data/lre/natural_lre_vs_behavior.csv"
OUT_DIR = Path("experiments/tables")


def format_pct(x: float) -> str:
    """Format a probability as percentage with 1 decimal."""
    return f"{100.0 * x:.1f}"


def build_behavior_table(df: pd.DataFrame) -> str:
    """
    Build a LaTeX table for refusal / hallucination rates
    for all models and relations.
    Rows: relations.
    Columns: for each model -> hallucination %, refusal %.
    """
    # We keep a consistent relation order
    rel_order = ["father", "instrument", "sport", "company_ceo"]
    df = df.copy()
    df["relation"] = pd.Categorical(df["relation"], categories=rel_order, ordered=True)
    df = df.sort_values(["relation", "model_key"])

    models = sorted(df["model_key"].unique())

    # Header row
    header = ["Relation"]
    for m in models:
        header.append(f"Halluc.~({m})")
        header.append(f"Refusal~({m})")

    lines = []
    lines.append("\\begin{tabular}{l" + "cc" * len(models) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for rel in rel_order:
        sub = df[df["relation"] == rel]
        if sub.empty:
            continue
        row = [rel.replace("_", "\\_")]
        # Ensure same model order for each row
        for m in models:
            s = sub[sub["model_key"] == m]
            if s.empty:
                row.extend(["--", "--"])
                continue
            halluc = format_pct(float(s["halluc_rate"].iloc[0]))
            refusal = format_pct(float(s["refusal_rate"].iloc[0]))
            row.extend([halluc, refusal])
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    tex_behavior = build_behavior_table(df)
    out_path = OUT_DIR / "relpanel_behavior.tex"
    out_path.write_text(tex_behavior, encoding="utf-8")

    print(f"[tables] Wrote LaTeX behavior table to {out_path}")


if __name__ == "__main__":
    main()
