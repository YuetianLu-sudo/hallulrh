import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mounts/work/yuetian_lu/hallulrh")
os.chdir(ROOT)

NAT_EXP = Path("runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209")
PJSON = NAT_EXP / "inputs_all47_step5/llama3_1_8b_instruct.all47.for_step5.with_gold.jsonl"

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_KEY = "llama3_1_8b_instruct"

PROMPT_VARIANT = os.environ.get("LLAMA_SWEEP_PROMPT_VARIANT", "plain")
USE_CHAT = PROMPT_VARIANT == "chat"

MIN_FREE_MIB = int(os.environ.get("MIN_FREE_MIB", "30000"))
MAX_UTIL = int(os.environ.get("MAX_UTIL", "100"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "30"))

# Broader than the previous quick sensitivity check, but still manageable.
SUBJECT_LAYERS = [6, 8, 10, 12, 14, 16, 18, 20, 22]
OBJECT_LAYERS = [22, 24, 26, 28, 30, 31]

def query_gpus():
    txt = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in txt.strip().splitlines():
        idx, free, used, util = [x.strip() for x in line.split(",")]
        rows.append({"idx": int(idx), "free": int(free), "used": int(used), "util": int(util)})
    rows.sort(key=lambda x: (-x["free"], x["util"], x["idx"]))
    return rows

def main():
    if not PJSON.exists():
        raise FileNotFoundError(PJSON)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(f"runs/experiments/llama_all47_layer_sweep_{PROMPT_VARIANT}_{ts}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    Path(f"runs/llama_all47_layer_sweep_{PROMPT_VARIANT}.latest").write_text(str(out), encoding="utf-8")

    print(f"[OUT] {out}", flush=True)
    print(f"[PROMPT_VARIANT] {PROMPT_VARIANT} USE_CHAT={USE_CHAT}", flush=True)
    print(f"[POLICY] MIN_FREE_MIB={MIN_FREE_MIB}, MAX_UTIL={MAX_UTIL}, POLL_SECONDS={POLL_SECONDS}", flush=True)

    jobs = []
    for s in SUBJECT_LAYERS:
        for o in OBJECT_LAYERS:
            if o > s:
                jobs.append((s, o))

    print(f"[jobs] {len(jobs)} layer pairs", flush=True)

    queue = jobs[:]
    running = {}
    failed = []

    while queue or running:
        done = []
        for key, item in list(running.items()):
            code = item["proc"].poll()
            if code is not None:
                item["log_fh"].close()
                done.append(key)
                if code == 0:
                    print(f"[done] {key} on GPU {item['gpu']}", flush=True)
                else:
                    print(f"[fail] {key} on GPU {item['gpu']} exit={code}", flush=True)
                    failed.append(key)

        for key in done:
            running.pop(key, None)

        if failed:
            raise SystemExit(f"[FAIL] workers failed: {failed}")

        used = {v["gpu"] for v in running.values()}
        candidates = [
            g for g in query_gpus()
            if g["idx"] not in used and g["free"] >= MIN_FREE_MIB and g["util"] <= MAX_UTIL
        ]

        while queue and candidates:
            s, o = queue.pop(0)
            key = f"s{s}_o{o}"
            gpu = candidates.pop(0)["idx"]
            model_out = out / key
            model_out.mkdir(parents=True, exist_ok=True)

            log_path = out / "logs" / f"{key}.log"
            log_fh = open(log_path, "w", encoding="utf-8")

            cmd = [
                sys.executable,
                "scripts/lre_step5_deltacos_gold_per_triple.py",
                "--model_id", MODEL_ID,
                "--model_key", MODEL_KEY,
                "--prompts_jsonl", str(PJSON),
                "--out_dir", str(model_out),
                "--train_frac", "0.75",
                "--seed", "0",
                "--batch_size", "8",
                "--subject_layer", str(s),
                "--object_layer", str(o),
            ]
            if USE_CHAT:
                cmd.append("--use_chat_template")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
            running[key] = {"proc": proc, "gpu": gpu, "log_fh": log_fh}
            print(f"[launch] {key} -> GPU {gpu}, pid={proc.pid}", flush=True)

        if queue:
            gstr = ", ".join(f"{g['idx']}:free={g['free']}MiB,util={g['util']}%" for g in query_gpus())
            rstr = ", ".join(f"{k}@{v['gpu']}" for k, v in running.items())
            print(f"[wait] queued={len(queue)}; running={rstr}; gpus={gstr}", flush=True)

        time.sleep(POLL_SECONDS)

    print(f"[done] OUT={out}", flush=True)

if __name__ == "__main__":
    main()
