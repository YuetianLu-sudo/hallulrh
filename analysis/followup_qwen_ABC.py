import os, re, math, shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

TS = datetime.now().strftime("%Y%m%d")

def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

def pick_col(cols, candidates, required=True):
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise KeyError(f"Missing columns. Tried: {candidates}. Have: {sorted(cols)[:50]} ...")
    return None

def normalize_path(p: Path) -> str:
    return str(p).replace("\\", "/")

def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

def find_join_csv() -> Path:
    env = os.environ.get("JOIN_CSV", "").strip()
    if env:
        p = Path(env)
        if not p.exists():
            raise SystemExit(f"ERROR: JOIN_CSV is set but file not found: {p}")
        return p

    need_any_model = {"model_name", "model_key", "model", "model_id"}
    need_cols = {"relation", "cos_improvement", "halluc_rate"}
    candidates = []
    for p in Path(".").rglob("*.csv"):
        sp = normalize_path(p)
        if "/data/judge/" in sp:
            continue
        if "/.venv/" in sp or "/site-packages/" in sp:
            continue
        try:
            df0 = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        cols = set(df0.columns)
        if need_cols.issubset(cols) and (cols & need_any_model):
            candidates.append(p)
    if not candidates:
        raise SystemExit(
            "ERROR: cannot auto-find JOIN CSV.\n"
            "Fix: export JOIN_CSV=/path/to/your_join.csv and rerun.\n"
            "The join CSV must contain at least: relation, cos_improvement, halluc_rate, and a model column."
        )
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]

def find_qwen_judge_csv() -> Path:
    env = os.environ.get("QWEN_CSV", "").strip()
    if env:
        p = Path(env)
        if not p.exists():
            raise SystemExit(f"ERROR: QWEN_CSV is set but file not found: {p}")
        return p

    default = Path("data/judge/relpanel_q/qwen2_5_7b_instruct_relpanel_with_judge.csv")
    if default.exists():
        return default

    cands = list(Path(".").rglob("qwen2_5_7b_instruct*with_judge*.csv"))
    if cands:
        cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return cands[0]

    raise SystemExit(
        "ERROR: cannot find Qwen judge csv.\n"
        "Expected: data/judge/relpanel_q/qwen2_5_7b_instruct_relpanel_with_judge.csv\n"
        "Fix: export QWEN_CSV=/path/to/qwen_with_judge.csv and rerun."
    )

def find_audit_dir_and_unpack_if_needed(out_dir: Path) -> Optional[Path]:
    # prefer existing txt dir
    for p in [Path("analysis/qwen_outlier_audit"), Path("qwen_outlier_audit")]:
        if p.exists() and list(p.rglob("*.txt")):
            return p

    # try zip
    zip_path = None
    for p in Path(".").rglob("qwen_outlier_audit.zip"):
        zip_path = p
        break
    if zip_path is None:
        return None

    unpack_dir = out_dir / f"qwen_outlier_audit_unzipped_{TS}"
    if unpack_dir.exists():
        return unpack_dir
    unpack_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), str(unpack_dir))
    return unpack_dir

def parse_qa_txt(p: Path, k: int = 2):
    """Parse audit txt format: Q/A/Reason blocks."""
    lines = p.read_text(errors="ignore").splitlines()
    out = []
    cur = {}
    for line in lines:
        if line.startswith("Q:"):
            if cur:
                out.append(cur)
                cur = {}
            cur["Q"] = line[2:].strip()
        elif line.startswith("A:"):
            cur["A"] = line[2:].strip()
        elif line.startswith("Reason:"):
            cur["Reason"] = line[7:].strip()
    if cur:
        out.append(cur)
    return out[:k]

def df_to_markdown(df: pd.DataFrame) -> str:
    """Dependency-free markdown table (no tabulate required)."""
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    def fmt(x):
        if isinstance(x, float):
            if math.isnan(x):
                return "NA"
            return f"{x:.3f}"
        return str(x)
    rows = [[fmt(v) for v in row] for row in df[cols].itertuples(index=False, name=None)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([header, sep, body])

def get_task_value(proxy: pd.DataFrame, task: str, col: str):
    s = proxy.loc[proxy["task"] == task, col]
    if len(s) == 0:
        return None
    v = s.values[0]
    try:
        return float(v)
    except Exception:
        return None

def main():
    out_dir = Path("analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load join CSV
    # -----------------------------
    join_path = find_join_csv()
    print(f"[join] {join_path}")
    df = pd.read_csv(join_path)

    model_col = pick_col(df.columns, ["model_key", "model_name", "model", "model_id"])
    rel_col = pick_col(df.columns, ["relation"])
    cos_col = pick_col(df.columns, ["cos_improvement", "delta_cos", "dcos", "lre_cos_improvement"])
    halluc_rate_col = pick_col(df.columns, ["halluc_rate", "hallucination_rate", "halluc_rate_mean"])
    refusal_rate_col = pick_col(df.columns, ["refusal_rate"], required=False)

    total_col = pick_col(df.columns, ["total", "n_total", "count", "n"], required=False)
    halluc_cnt_col = pick_col(df.columns, ["hallucination", "n_hallucination", "halluc_cnt", "halluc"], required=False)
    refusal_cnt_col = pick_col(df.columns, ["refusal", "n_refusal", "refusal_cnt"], required=False)
    other_cnt_col = pick_col(df.columns, ["other", "n_other"], required=False)

    # derive missing refusal_rate if needed
    if refusal_rate_col is None:
        if total_col and refusal_cnt_col:
            df["refusal_rate"] = df[refusal_cnt_col] / df[total_col]
            refusal_rate_col = "refusal_rate"
        else:
            raise SystemExit("ERROR: join CSV has no refusal_rate and no (refusal,total) counts to derive it.")

    # other rate (optional)
    if other_cnt_col and total_col:
        df["other_rate"] = df[other_cnt_col] / df[total_col]
        other_rate_col = "other_rate"
    else:
        other_rate_col = None

    # -----------------------------
    # A) Refusal-focused plots
    #   A1: 3-way breakdown
    #   A2: refusal-only (+ Wilson CI if counts exist)
    # -----------------------------
    relation_order = ["father", "instrument", "sport", "company_ceo", "company_hq", "country_language"]
    def rel_sort_key(r):
        return relation_order.index(r) if r in relation_order else 999

    for model, g in df.groupby(model_col):
        g = g.copy()
        g["__order"] = g[rel_col].astype(str).apply(rel_sort_key)
        g = g.sort_values("__order")

        refusal = g[refusal_rate_col].astype(float).clip(0, 1)
        halluc = g[halluc_rate_col].astype(float).clip(0, 1)
        other = g[other_rate_col].astype(float).clip(0, 1) if other_rate_col else 0.0
        non_halluc_non_refusal = (1.0 - refusal - halluc - other).clip(0, 1)

        y = np.arange(len(g))

        # A1: 3-way breakdown
        fig, ax = plt.subplots(figsize=(10, 4 + 0.35 * len(g)))
        ax.barh(y, non_halluc_non_refusal * 100, label="Non-hallucination (non-refusal)")
        ax.barh(y, halluc * 100, left=non_halluc_non_refusal * 100, label="Hallucination")
        ax.barh(y, refusal * 100, left=(non_halluc_non_refusal + halluc) * 100, label="Refusal")
        ax.set_yticks(y)
        ax.set_yticklabels(g[rel_col].astype(str))
        ax.set_xlim(0, 100)
        ax.set_xlabel("Rate (%)")
        ax.set_title(f"{model}: response breakdown (A1, {TS})")
        ax.legend(loc="lower right", frameon=True)
        fig.tight_layout()
        f1 = out_dir / f"fig_A1_behavior3way_{safe_filename(str(model))}_{TS}.png"
        fig.savefig(f1, dpi=200)
        plt.close(fig)

        # A2: refusal-only
        fig, ax = plt.subplots(figsize=(10, 4 + 0.25 * len(g)))
        x = refusal * 100
        if total_col and refusal_cnt_col:
            lows, highs = [], []
            for k, n in zip(g[refusal_cnt_col].astype(int), g[total_col].astype(int)):
                lo, hi = wilson_ci(int(k), int(n))
                lows.append(lo)
                highs.append(hi)
            lows = np.array(lows)
            highs = np.array(highs)
            xerr = np.vstack([(refusal.values - lows) * 100, (highs - refusal.values) * 100])
            ax.barh(y, x, xerr=xerr, capsize=3)
        else:
            ax.barh(y, x)

        ax.set_yticks(y)
        ax.set_yticklabels(g[rel_col].astype(str))
        ax.set_xlim(0, 100)
        ax.set_xlabel("Refusal rate (%)")
        ax.set_title(f"{model}: refusal rate (A2, {TS})")
        fig.tight_layout()
        f2 = out_dir / f"fig_A2_refusal_{safe_filename(str(model))}_{TS}.png"
        fig.savefig(f2, dpi=200)
        plt.close(fig)

        print(f"[write] {f1}")
        print(f"[write] {f2}")

    # -----------------------------
    # B) Qwen entity recognizability proxy
    #   - 真实国家匹配率（pycountry）: data-level recognizability
    #   - refusal 中“实体不被承认/不存在”话术占比: model-level recognizability
    # -----------------------------
    qwen_path = find_qwen_judge_csv()
    print(f"[qwen] {qwen_path}")
    dfq = pd.read_csv(qwen_path)

    dfq["judge_label_norm"] = dfq["judge_label"].astype(str).str.strip().str.lower()

    qcol = pick_col(dfq.columns, ["question", "prompt", "input"], required=True)
    acol = pick_col(dfq.columns, ["answer", "response", "output"], required=True)
    task_col = pick_col(dfq.columns, ["task"], required=True)

    unrec_re = re.compile(
        r"(not a (?:recognized|real|valid)|fictional|does(?:n't)? exist|does not exist|"
        r"no such (?:country|place|entity)|not (?:an|a) (?:existing|established|known)|"
        r"unable to find|could not find|cannot find|can't find|not found)",
        re.IGNORECASE,
    )
    dfq["unrecognized_phrase"] = dfq[acol].astype(str).apply(lambda s: bool(unrec_re.search(s)))

    def extract_country(q: str) -> Optional[str]:
        q = str(q)
        q = re.sub(r"^\s*Q:\s*", "", q, flags=re.IGNORECASE).strip()
        q = re.sub(r"\s+", " ", q)

        m = re.search(r"(?i)what is\s+(.+?)'s\s+official language", q)
        if m:
            return m.group(1).strip()

        m = re.search(r"(?i)(?:official language of|language of|speak in)\s+(.+?)[\?\.]?\s*$", q)
        if m:
            return m.group(1).strip()

        m = re.search(r"(?i)(?:of|in)\s+(.+?)[\?\.]?\s*$", q)
        if m:
            return m.group(1).strip()

        return None

    try:
        import pycountry  # type: ignore
    except Exception:
        pycountry = None
        print("[warn] pycountry not installed -> real_country_match_rate will be NA (run: python -m pip install pycountry)")

    def is_real_country(name: str) -> bool:
        if not name or pycountry is None:
            return False
        try:
            pycountry.countries.search_fuzzy(name)
            return True
        except Exception:
            return False

    rows = []
    for task, g in dfq.groupby(task_col):
        n = len(g)
        n_ref = int((g["judge_label_norm"] == "refusal").sum())
        n_hal = int((g["judge_label_norm"] == "hallucination").sum())
        rec = {
            "task": task,
            "n": n,
            "refusal_rate": n_ref / n if n else np.nan,
            "halluc_rate": n_hal / n if n else np.nan,
            "refusal_unrecognized_phrase_share": float(
                g.loc[g["judge_label_norm"] == "refusal", "unrecognized_phrase"].mean()
            )
            if n_ref > 0
            else np.nan,
        }

        if task == "country_language":
            ents = g[qcol].apply(extract_country)
            rec["country_entity_parse_success_rate"] = float(ents.notna().mean())
            if pycountry is not None and ents.notna().any():
                rec["real_country_match_rate"] = float(ents.dropna().apply(is_real_country).mean())
            else:
                rec["real_country_match_rate"] = np.nan

        rows.append(rec)

    proxy = pd.DataFrame(rows).sort_values("task")
    out_proxy = out_dir / f"table_B_qwen_entity_proxy_{TS}.csv"
    proxy.to_csv(out_proxy, index=False)
    print(f"[write] {out_proxy}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(proxy["task"], proxy["refusal_rate"] * 100)
    ax.set_xlabel("Refusal rate (%)")
    ax.set_title(f"Qwen2.5-7B-Instruct: refusal rate by task (B1, {TS})")
    fig.tight_layout()
    f_b1 = out_dir / f"fig_B1_qwen_refusal_by_task_{TS}.png"
    fig.savefig(f_b1, dpi=200)
    plt.close(fig)
    print(f"[write] {f_b1}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(proxy["task"], proxy["refusal_unrecognized_phrase_share"] * 100)
    ax.set_xlabel("Share within refusals (%)")
    ax.set_title(f"Qwen2.5-7B-Instruct: 'entity unrecognized' refusals (B2, {TS})")
    fig.tight_layout()
    f_b2 = out_dir / f"fig_B2_qwen_unrecognized_in_refusals_{TS}.png"
    fig.savefig(f_b2, dpi=200)
    plt.close(fig)
    print(f"[write] {f_b2}")

    # -----------------------------
    # C) Mark Qwen as outlier
    #   - pooled scatter highlight Qwen
    #   - correlation with vs without Qwen
    #   - qualitative appendix from zip/txt
    # -----------------------------
    x_all = df[cos_col].astype(float).values
    y_all = df[halluc_rate_col].astype(float).values
    rho_all, p_all = spearmanr(x_all, y_all)

    is_qwen = df[model_col].astype(str).str.lower().str.contains("qwen")
    df_noq = df.loc[~is_qwen].copy()
    rho_noq, p_noq = spearmanr(df_noq[cos_col].astype(float).values, df_noq[halluc_rate_col].astype(float).values)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for model, g in df.groupby(model_col):
        xs = g[cos_col].astype(float).values
        ys = g[halluc_rate_col].astype(float).values
        if str(model).lower().find("qwen") >= 0:
            ax.scatter(xs, ys, label=str(model) + " (outlier)", s=140, edgecolors="black", linewidths=1.5)
        else:
            ax.scatter(xs, ys, label=str(model), s=90, alpha=0.9)

    ax.set_xlabel("LRE cosine improvement (Δcos)")
    ax.set_ylabel("Hallucination rate")
    ax.set_title(
        f"Pooled Δcos vs hallucination (C, {TS})\n"
        f"Spearman ρ={rho_all:.3f} (all), ρ={rho_noq:.3f} (no Qwen)"
    )
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    f_c = out_dir / f"fig_C_pooled_scatter_outlier_{TS}.png"
    fig.savefig(f_c, dpi=200)
    plt.close(fig)
    print(f"[write] {f_c}")

    audit_dir = find_audit_dir_and_unpack_if_needed(out_dir)

    md_lines = []
    md_lines.append(f"# Qwen2.5-7B-Instruct as a strategy outlier ({TS})")
    md_lines.append("")
    md_lines.append("## Quantitative (from join CSV)")
    md_lines.append(f"- Pooled Spearman(Δcos, hallucination_rate): ρ={rho_all:.3f}, p={p_all:.3g}, N={len(df)}")
    md_lines.append(f"- Excluding Qwen: ρ={rho_noq:.3f}, p={p_noq:.3g}, N={len(df_noq)}")
    md_lines.append("")
    md_lines.append("## Qwen behavior by task (relpanel_q judge CSV)")
    md_lines.append("")
    md_lines.append(df_to_markdown(proxy))
    md_lines.append("")
    md_lines.append("## Qualitative examples (from qwen_outlier_audit.zip)")
    if audit_dir is None:
        md_lines.append("- (qwen_outlier_audit.zip not found; put it under repo root or analysis/ and rerun.)")
    else:
        txts = list(audit_dir.rglob("*.txt"))
        want = [
            ("country_language", "refusal"),
            ("country_language", "hallucination"),
            ("company_hq", "refusal"),
            ("company_hq", "hallucination"),
            ("company_ceo", "refusal"),
            ("company_ceo", "hallucination"),
            ("sport", "refusal"),
            ("sport", "hallucination"),
        ]
        for task, lab in want:
            cand = None
            for p in txts:
                name = p.name.lower()
                if name.startswith(task) and lab in name:
                    cand = p
                    break
            if cand is None:
                continue

            md_lines.append(f"### {task} / {lab}")
            for ex in parse_qa_txt(cand, k=2):
                q = ex.get("Q", "")
                a = ex.get("A", "")
                r = ex.get("Reason", "")
                md_lines.append(f"> Q: {q}")
                md_lines.append(f"> A: {a}")
                if r:
                    md_lines.append(f"> Judge: {r}")
                md_lines.append(">")
            md_lines.append("")

    out_md = out_dir / f"text_C_qwen_outlier_appendix_{TS}.md"
    out_md.write_text("\n".join(md_lines))
    print(f"[write] {out_md}")

    cl_ref = get_task_value(proxy, "country_language", "refusal_rate")
    ceo_ref = get_task_value(proxy, "company_ceo", "refusal_rate")
    cl_ref_s = "NA" if cl_ref is None else f"{cl_ref:.3f}"
    ceo_ref_s = "NA" if ceo_ref is None else f"{ceo_ref:.3f}"

    main_par = (
        f"Qwen2.5-7B-Instruct behaves as a strategy outlier: it frequently refuses instead of answering "
        f"(e.g., country_language refusal_rate={cl_ref_s}, company_ceo refusal_rate={ceo_ref_s}). "
        f"As a result, its raw hallucination rate is not directly comparable to models that answer more often. "
        f"In pooled analysis, Spearman ρ between Δcos and hallucination_rate is {rho_all:.3f} (all models) "
        f"and {rho_noq:.3f} when excluding Qwen."
    )
    out_main = out_dir / f"text_C_qwen_outlier_main_{TS}.txt"
    out_main.write_text(main_par + "\n")
    print(f"[write] {out_main}")

if __name__ == "__main__":
    main()
