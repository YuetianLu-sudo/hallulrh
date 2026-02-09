#!/usr/bin/env python3
import argparse, os, glob, json
from collections import defaultdict
import pandas as pd
import numpy as np

def norm_label(rec):
    for k in ["lm_judge_label", "judge_label", "label"]:
        v = rec.get(k)
        if v is not None and str(v).strip():
            s = str(v).strip().upper()
            if "REFUS" in s: return "REFUSAL"
            if "HALL" in s: return "HALLUCINATION"
            if "CORR" in s: return "CORRECT"
            return s
    return "UNKNOWN"

def norm_key(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")

def pearson_r(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 2: return np.nan
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt((xc @ xc) * (yc @ yc))
    if denom <= 0: return np.nan
    return float((xc @ yc) / denom)

def spearman_rho(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 2: return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return pearson_r(rx, ry)

def weighted_corr(x, y, w):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[m]; y = y[m]; w = w[m]
    if len(x) < 2: return np.nan
    w = w / w.sum()
    mx = (w * x).sum()
    my = (w * y).sum()
    cov = (w * (x - mx) * (y - my)).sum()
    vx = (w * (x - mx) ** 2).sum()
    vy = (w * (y - my) ** 2).sum()
    denom = np.sqrt(vx * vy)
    if denom <= 0: return np.nan
    return float(cov / denom)

def find_deltacos_csv(root_candidates=("runs", "data")):
    patterns = []
    for root in root_candidates:
        patterns += [
            f"{root}/**/*deltacos*.csv",
            f"{root}/**/*delta_cos*.csv",
            f"{root}/**/*cos_improvement*.csv",
        ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))

    best = None
    best_score = (-1, -1)  # (n_rel, n_models)
    best_cols = None

    for fp in files:
        try:
            df0 = pd.read_csv(fp, nrows=200)
        except Exception:
            continue
        cols = set(df0.columns)

        # identify columns
        rel_col = None
        for c in ["relation_key", "relation", "task", "rel"]:
            if c in cols:
                rel_col = c
                break
        model_col = None
        for c in ["model_key", "model", "model_name"]:
            if c in cols:
                model_col = c
                break
        val_col = None
        for c in ["cos_improvement", "delta_cos", "deltacos", "DeltaCos", "Δcos"]:
            if c in cols:
                val_col = c
                break

        if not (rel_col and model_col and val_col):
            continue

        n_rel = df0[rel_col].astype(str).nunique()
        n_models = df0[model_col].astype(str).nunique()

        score = (n_rel, n_models)
        if score > best_score:
            best_score = score
            best = fp
            best_cols = (rel_col, model_col, val_col)

    return best, best_cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--deltacos", default=None, help="Optional path to delta-cos CSV. If omitted, auto-searches.")
    ap.add_argument("--min-n-test", type=int, default=10, help="If deltacos has n_test, filter relations with n_test < this.")
    ap.add_argument("--n-perm", type=int, default=20000, help="Monte Carlo permutations for p-values (per model per metric).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -------- load judged jsonl (all models) --------
    paths = sorted(glob.glob(os.path.join(args.judge_dir, "*.judged.jsonl")))
    if not paths:
        raise RuntimeError(f"No *.judged.jsonl under {args.judge_dir}")

    rows = []
    for p in paths:
        for line in open(p, "r", encoding="utf-8"):
            rec = json.loads(line)
            model_key = rec.get("model_key") or os.path.basename(p).split(".")[0]
            relation_key = rec.get("relation_key")
            if not relation_key:
                continue
            rows.append({
                "model_key": norm_key(model_key),
                "relation_key": norm_key(relation_key),
                "label": norm_label(rec),
                "domain": rec.get("domain", ""),
                "relation_group": rec.get("relation_group", ""),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "lre3way_long_all47.csv"), index=False)

    # -------- aggregate --------
    g = df.groupby(["model_key", "relation_key"])
    agg = g["label"].value_counts().unstack(fill_value=0).reset_index()
    for col in ["REFUSAL", "CORRECT", "HALLUCINATION", "UNKNOWN"]:
        if col not in agg.columns:
            agg[col] = 0
    agg["n_total"] = agg[["REFUSAL", "CORRECT", "HALLUCINATION", "UNKNOWN"]].sum(axis=1)
    agg["n_refusal"] = agg["REFUSAL"]
    agg["n_correct"] = agg["CORRECT"]
    agg["n_hall"] = agg["HALLUCINATION"]
    agg["answered_n"] = agg["n_correct"] + agg["n_hall"]
    agg["noncorrect_n"] = agg["n_refusal"] + agg["n_hall"]

    # Fig2 original (answered-only)
    agg["hall_rate_answered"] = np.where(agg["answered_n"] > 0, agg["n_hall"] / agg["answered_n"], np.nan)
    # Fig1-style on natural (exclude correct)
    agg["hall_rate_noncorrect"] = np.where(agg["noncorrect_n"] > 0, agg["n_hall"] / agg["noncorrect_n"], np.nan)

    out_summary = os.path.join(args.outdir, "lre3way_summary_by_relation_all47.csv")
    agg.to_csv(out_summary, index=False)
    print(f"[done] wrote: {out_summary}")

    # -------- load deltacos and merge --------
    deltacos_path = args.deltacos
    cols = None
    if deltacos_path is None:
        deltacos_path, cols = find_deltacos_csv()
        if deltacos_path:
            print(f"[info] auto-selected deltacos: {deltacos_path}")
        else:
            print("[warn] could not auto-find deltacos CSV. You can still use summary CSV without correlation.")
            return

    ddf = pd.read_csv(deltacos_path)
    cols = cols or None

    # guess cols if auto info missing
    cols_set = set(ddf.columns)
    rel_col = None
    for c in ["relation_key", "relation", "task", "rel"]:
        if c in cols_set: rel_col = c; break
    model_col = None
    for c in ["model_key", "model", "model_name"]:
        if c in cols_set: model_col = c; break
    val_col = None
    for c in ["cos_improvement", "delta_cos", "deltacos", "DeltaCos", "Δcos"]:
        if c in cols_set: val_col = c; break
    if not (rel_col and model_col and val_col):
        raise RuntimeError(f"deltacos CSV missing required columns. Found columns: {sorted(cols_set)[:30]}...")

    ddf = ddf.copy()
    ddf["model_key"] = ddf[model_col].astype(str).map(norm_key)
    ddf["relation_key"] = ddf[rel_col].astype(str).map(norm_key)
    ddf["cos_improvement"] = pd.to_numeric(ddf[val_col], errors="coerce")

    # weight column if exists
    wcol = None
    for c in ["n_test", "n_pairs", "n_examples", "count"]:
        if c in ddf.columns:
            wcol = c
            break
    if wcol:
        ddf["weight"] = pd.to_numeric(ddf[wcol], errors="coerce")
    else:
        ddf["weight"] = np.nan

    merged = agg.merge(ddf[["model_key", "relation_key", "cos_improvement", "weight"]], on=["model_key", "relation_key"], how="left")
    merged_path = os.path.join(args.outdir, "lre3way_behavior_plus_deltacos_all47.csv")
    merged.to_csv(merged_path, index=False)
    print(f"[done] wrote: {merged_path}")

    # Optional filter if n_test exists (to match paper's "remove test <= 10" idea)
    merged_f = merged.copy()
    if "weight" in merged_f.columns and np.isfinite(merged_f["weight"]).any():
        merged_f = merged_f[merged_f["weight"].fillna(0) >= args.min_n_test].copy()
        filt_path = os.path.join(args.outdir, f"lre3way_behavior_plus_deltacos_min{args.min_n_test}.csv")
        merged_f.to_csv(filt_path, index=False)
        print(f"[done] wrote: {filt_path} (filtered by weight>={args.min_n_test})")

    # -------- correlation summaries (per model) --------
    def perm_p(x, y, n_perm, seed=0):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]; y = y[m]
        n = len(x)
        if n < 3:
            return np.nan, np.nan
        rng = np.random.default_rng(seed)
        xc = x - x.mean()
        yc = y - y.mean()
        denom = np.sqrt((xc @ xc) * (yc @ yc))
        if denom <= 0:
            return np.nan, np.nan
        r_obs = float((xc @ yc) / denom)

        # permutation: denom_y invariant
        denom_x = np.sqrt(xc @ xc)
        denom_y = np.sqrt(yc @ yc)
        if denom_x <= 0 or denom_y <= 0:
            return np.nan, np.nan

        ge = 0
        abs_ge = 0
        for _ in range(n_perm):
            yp = rng.permutation(yc)
            r = float((xc @ yp) / (denom_x * denom_y))
            if r >= r_obs: ge += 1
            if abs(r) >= abs(r_obs): abs_ge += 1
        return ge / n_perm, abs_ge / n_perm

    corr_rows = []
    for df_use, tag in [(merged, "all47"), (merged_f, f"min{args.min_n_test}")]:
        for mk in sorted(df_use["model_key"].dropna().unique()):
            sub = df_use[df_use["model_key"] == mk].copy()

            # metrics to report
            for ycol in ["hall_rate_answered", "hall_rate_noncorrect"]:
                x = sub["cos_improvement"].to_numpy()
                y = sub[ycol].to_numpy()
                r = pearson_r(x, y)
                rho = spearman_rho(x, y)

                # weighted by deltacos weight if available; else NaN
                w = sub["weight"].to_numpy()
                r_w = weighted_corr(x, y, w) if np.isfinite(w).any() else np.nan

                p1, p2 = perm_p(x, y, n_perm=args.n_perm, seed=0)

                corr_rows.append({
                    "subset": tag,
                    "model_key": mk,
                    "y_metric": ycol,
                    "n_rel": int(np.isfinite(x).sum()),
                    "pearson_r": r,
                    "spearman_rho": rho,
                    "perm_p_one": p1,
                    "perm_p_two": p2,
                    "r_weighted_by_deltacos_n": r_w,
                })

    corr_df = pd.DataFrame(corr_rows)
    corr_path = os.path.join(args.outdir, "lre3way_corr_summary.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"[done] wrote: {corr_path}")

if __name__ == "__main__":
    main()
