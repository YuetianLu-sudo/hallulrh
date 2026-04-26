import gzip
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path("/mounts/work/yuetian_lu/hallulrh")
os.chdir(ROOT)

NAT_EXP = Path("runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209")
SYN_EXP = Path("runs/experiments/exp1_lre21_synth_20260119_211221")

MIN_FREE_MIB = int(os.environ.get("MIN_FREE_MIB", "30000"))
MAX_UTIL = int(os.environ.get("MAX_UTIL", "100"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "30"))

MODELS = [
    "gemma_7b_it",
    "llama3_1_8b_instruct",
    "mistral_7b_instruct",
    "qwen2_5_7b_instruct",
]

PROMPTS = {
    "gemma_7b_it": NAT_EXP / "inputs/gemma_7b_it.for_3way_judge..with_gold.jsonl",
    "llama3_1_8b_instruct": NAT_EXP / "inputs/llama3_1_8b_instruct.for_3way_judge..with_gold.jsonl",
    "mistral_7b_instruct": NAT_EXP / "inputs/mistral_7b_instruct.for_3way_judge..with_gold.jsonl",
    "qwen2_5_7b_instruct": NAT_EXP / "inputs/qwen2_5_7b_instruct.for_3way_judge..with_gold.jsonl",
}

SYN15 = [
    "company_ceo",
    "company_hq",
    "landmark_in_country",
    "landmark_on_continent",
    "person_father",
    "person_mother",
    "person_occupation",
    "person_plays_instrument",
    "person_plays_position_in_sport",
    "person_plays_pro_sport",
    "person_university",
    "product_by_company",
    "star_constellation",
    "superhero_archnemesis",
    "superhero_person",
]


def sh(cmd, **kwargs):
    return subprocess.check_output(cmd, text=True, **kwargs)


def latest_default_chat_base():
    cands = sorted(Path("runs/experiments").glob("lre_layer_prompt_sensitivity_multigpu_*/default_chat"))
    if not cands:
        raise FileNotFoundError(
            "Could not find default_chat Step5 base. Expected runs/experiments/lre_layer_prompt_sensitivity_multigpu_*/default_chat"
        )
    return cands[-1]


def query_gpus():
    txt = sh([
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    for line in txt.strip().splitlines():
        idx, free, used, util = [x.strip() for x in line.split(",")]
        rows.append({
            "idx": int(idx),
            "free": int(free),
            "used": int(used),
            "util": int(util),
        })
    rows.sort(key=lambda x: (-x["free"], x["util"], x["idx"]))
    return rows


def run_summary(out: Path):
    nat = pd.read_csv(NAT_EXP / "analysis_all47/lre3way_behavior_plus_deltacos_all47.csv")
    if "model_key" not in nat.columns and "model_name" in nat.columns:
        nat = nat.rename(columns={"model_name": "model_key"})
    if "relation_key" not in nat.columns and "relation" in nat.columns:
        nat = nat.rename(columns={"relation": "relation_key"})
    nat["acc_answered"] = 1.0 - nat["hall_rate_answered"]

    audit_dirs = sorted(Path("runs/audits").glob("synthal_collision_audit_*/recomputed/exp1_behavior_plus_deltacos_factual_clean.csv"))
    if audit_dirs:
        syn_path = audit_dirs[-1]
    else:
        syn_path = SYN_EXP / "analysis/exp1_behavior_plus_deltacos_factual.csv"

    syn = pd.read_csv(syn_path)
    if "model_key" not in syn.columns and "model_name" in syn.columns:
        syn = syn.rename(columns={"model_name": "model_key"})
    if "relation_key" not in syn.columns and "relation" in syn.columns:
        syn = syn.rename(columns={"relation": "relation_key"})

    aff_rows = []
    for mk in MODELS:
        p = out / mk / "relation_summary_affine.csv.gz"
        if not p.exists():
            raise FileNotFoundError(p)
        with gzip.open(p, "rt", encoding="utf-8") as f:
            df = pd.read_csv(f)
        df["model_key"] = mk
        df["delta_cos_affine"] = df["cos_improvement_affine"]
        aff_rows.append(df)
    aff = pd.concat(aff_rows, ignore_index=True)
    aff.to_csv(out / "affine_relation_summary_all_models.csv", index=False)

    syn_corr_rows = []
    for mk in MODELS:
        a = aff[(aff["model_key"] == mk) & (aff["relation_key"].isin(SYN15))][
            ["model_key", "relation_key", "delta_cos_affine"]
        ]
        s = syn[syn["model_key"] == mk][["model_key", "relation_key", "hall_rate"]]
        m = a.merge(s, on=["model_key", "relation_key"], how="inner")
        r, p = pearsonr(m["delta_cos_affine"], m["hall_rate"])
        rho, ps = spearmanr(m["delta_cos_affine"], m["hall_rate"])
        syn_corr_rows.append({
            "model_key": mk,
            "n_rel": len(m),
            "pearson_affine_vs_hall": r,
            "pearson_p": p,
            "spearman_affine_vs_hall": rho,
            "spearman_p": ps,
        })
    syn_corr = pd.DataFrame(syn_corr_rows)
    syn_corr.to_csv(out / "affine_synth15_corr.csv", index=False)

    nat_corr_rows = []
    for mk in MODELS:
        a = aff[aff["model_key"] == mk][["model_key", "relation_key", "delta_cos_affine"]]
        n = nat[(nat["model_key"] == mk) & (nat["n_test"] > 10)][
            ["model_key", "relation_key", "acc_answered", "hall_rate_answered", "cos_improvement"]
        ]
        m = a.merge(n, on=["model_key", "relation_key"], how="inner")
        r, p = pearsonr(m["delta_cos_affine"], m["acc_answered"])
        rho, ps = spearmanr(m["delta_cos_affine"], m["acc_answered"])
        nat_corr_rows.append({
            "model_key": mk,
            "n_rel": len(m),
            "pearson_affine_vs_acc_answered": r,
            "pearson_p": p,
            "spearman_affine_vs_acc_answered": rho,
            "spearman_p": ps,
        })
    nat_corr = pd.DataFrame(nat_corr_rows)
    nat_corr.to_csv(out / "affine_natural_accuracy_corr.csv", index=False)

    rank_rows = []
    for mk in MODELS:
        a = aff[aff["model_key"] == mk][["model_key", "relation_key", "delta_cos_affine"]]
        n = nat[nat["model_key"] == mk][["model_key", "relation_key", "cos_improvement"]]
        m = a.merge(n, on=["model_key", "relation_key"], how="inner")
        r, p = pearsonr(m["cos_improvement"], m["delta_cos_affine"])
        rho, ps = spearmanr(m["cos_improvement"], m["delta_cos_affine"])
        rank_rows.append({
            "model_key": mk,
            "n_rel": len(m),
            "pearson_trans_vs_affine": r,
            "pearson_p": p,
            "spearman_trans_vs_affine": rho,
            "spearman_p": ps,
        })
    rank = pd.DataFrame(rank_rows)
    rank.to_csv(out / "trans_vs_affine_rank_stability.csv", index=False)

    with open(out / "latex_snippets.txt", "w", encoding="utf-8") as f:
        f.write("% Synthetic 15-relation affine metric correlations\n")
        for _, r in syn_corr.iterrows():
            f.write(
                f"{r['model_key']} & {r['pearson_affine_vs_hall']:.3f} & "
                f"${r['pearson_p']:.2e}$ & {r['spearman_affine_vs_hall']:.3f} & "
                f"${r['spearman_p']:.2e}$ \\\\\n"
            )
        f.write("\n% Natural answered-case accuracy affine correlations\n")
        for _, r in nat_corr.iterrows():
            f.write(
                f"{r['model_key']} & {r['pearson_affine_vs_acc_answered']:.3f} & "
                f"${r['pearson_p']:.2e}$ & {r['spearman_affine_vs_acc_answered']:.3f} & "
                f"${r['spearman_p']:.2e}$ \\\\\n"
            )

    print("[write]", out / "affine_relation_summary_all_models.csv", flush=True)
    print("[write]", out / "affine_synth15_corr.csv", flush=True)
    print("[write]", out / "affine_natural_accuracy_corr.csv", flush=True)
    print("[write]", out / "trans_vs_affine_rank_stability.csv", flush=True)
    print("[write]", out / "latex_snippets.txt", flush=True)
    print("\n=== affine_synth15_corr ===", flush=True)
    print(syn_corr.to_string(index=False), flush=True)
    print("\n=== affine_natural_accuracy_corr ===", flush=True)
    print(nat_corr.to_string(index=False), flush=True)
    print("\n=== trans_vs_affine_rank_stability ===", flush=True)
    print(rank.to_string(index=False), flush=True)


def main():
    base_root = latest_default_chat_base()
    relset = NAT_EXP / "inputs/relation_set_all47.txt"
    if not relset.exists():
        raise FileNotFoundError(relset)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(f"runs/experiments/full_affine_main_metric_greedy_{ts}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    Path("runs/full_affine_main_metric.latest").write_text(str(out), encoding="utf-8")

    print(f"[BASE_ROOT] {base_root}", flush=True)
    print(f"[RELSET]    {relset}", flush=True)
    print(f"[OUT]       {out}", flush=True)
    print(f"[POLICY]    MIN_FREE_MIB={MIN_FREE_MIB}, MAX_UTIL={MAX_UTIL}, POLL_SECONDS={POLL_SECONDS}", flush=True)

    for mk in MODELS:
        if not PROMPTS[mk].exists():
            raise FileNotFoundError(PROMPTS[mk])
        if not (base_root / mk / "relation_summary.csv.gz").exists():
            raise FileNotFoundError(base_root / mk / "relation_summary.csv.gz")

    queue = MODELS[:]
    running = {}
    failed = []

    while queue or running:
        finished = []
        for mk, item in running.items():
            p = item["proc"]
            code = p.poll()
            if code is not None:
                log_fh = item["log_fh"]
                log_fh.close()
                finished.append(mk)
                if code == 0:
                    print(f"[done] {mk} on GPU {item['gpu']}", flush=True)
                else:
                    print(f"[fail] {mk} on GPU {item['gpu']} exit_code={code}", flush=True)
                    failed.append(mk)

        for mk in finished:
            running.pop(mk, None)

        if failed:
            print("[FAIL] stopping because at least one worker failed:", failed, flush=True)
            sys.exit(1)

        used_by_us = {item["gpu"] for item in running.values()}
        gpus = query_gpus()
        candidates = [
            g for g in gpus
            if g["idx"] not in used_by_us
            and g["free"] >= MIN_FREE_MIB
            and g["util"] <= MAX_UTIL
        ]

        while queue and candidates:
            mk = queue.pop(0)
            gpu = candidates.pop(0)["idx"]
            model_out = out / mk
            model_out.mkdir(parents=True, exist_ok=True)

            log_path = out / "logs" / f"{mk}.log"
            log_fh = open(log_path, "w", encoding="utf-8")

            cmd = [
                sys.executable,
                "scripts/lre_step5_deltacos_gold_affine_from_base.py",
                "--prompts", str(PROMPTS[mk]),
                "--base-step5-dir", str(base_root / mk),
                "--relation-set", str(relset),
                "--model-key", mk,
                "--outdir", str(model_out),
                "--ridge-lambda", "1e-2",
                "--batch-size", "8",
                "--chat-template", "auto",
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
                env=env,
            )

            running[mk] = {
                "proc": proc,
                "gpu": gpu,
                "log_fh": log_fh,
                "log_path": str(log_path),
            }
            (out / "logs" / f"{mk}.pid").write_text(str(proc.pid), encoding="utf-8")
            print(f"[launch] {mk} -> GPU {gpu}, pid={proc.pid}, log={log_path}", flush=True)

        if queue:
            gpu_state = ", ".join([f"{g['idx']}:free={g['free']}MiB,util={g['util']}%" for g in gpus])
            run_state = ", ".join([f"{mk}@{it['gpu']}" for mk, it in running.items()])
            print(f"[wait] queued={queue}; running={run_state}; gpus={gpu_state}", flush=True)

        time.sleep(POLL_SECONDS)

    run_summary(out)
    print(f"[done] OUT={out}", flush=True)


if __name__ == "__main__":
    main()
