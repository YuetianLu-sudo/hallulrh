import os
import subprocess
import sys
import time
from pathlib import Path

from transformers import AutoConfig

ROOT = Path("/mounts/work/yuetian_lu/hallulrh")
os.chdir(ROOT)

NAT_EXP = Path("runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209")
INPUT_DIR = NAT_EXP / "inputs_all47_step5"

RUN_VARIANT = os.environ.get("RUN_VARIANT", "plain")
USE_CHAT = os.environ.get("USE_CHAT_TEMPLATE", "0") == "1"

MIN_FREE_MIB = int(os.environ.get("MIN_FREE_MIB", "30000"))
MAX_UTIL = int(os.environ.get("MAX_UTIL", "100"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "30"))

MODELS = [
    "gemma_7b_it",
    "llama3_1_8b_instruct",
    "mistral_7b_instruct",
    "qwen2_5_7b_instruct",
]

MODEL_ID = {
    "gemma_7b_it": "google/gemma-7b-it",
    "llama3_1_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral_7b_instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2_5_7b_instruct": "Qwen/Qwen2.5-7B-Instruct",
}


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
        rows.append(
            {
                "idx": int(idx),
                "free": int(free),
                "used": int(used),
                "util": int(util),
            }
        )
    rows.sort(key=lambda x: (-x["free"], x["util"], x["idx"]))
    return rows


def get_layers(model_id):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    L = int(getattr(cfg, "num_hidden_layers"))
    subject_layer = L // 2
    object_layer = max(subject_layer + 2, L - 2)
    return subject_layer, object_layer


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(f"runs/experiments/step5_all47_translation_{RUN_VARIANT}_{ts}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    Path(f"runs/step5_all47_translation_{RUN_VARIANT}.latest").write_text(
        str(out), encoding="utf-8"
    )

    print(f"[OUT] {out}", flush=True)
    print(f"[VARIANT] {RUN_VARIANT} USE_CHAT={USE_CHAT}", flush=True)
    print(
        f"[POLICY] MIN_FREE_MIB={MIN_FREE_MIB}, MAX_UTIL={MAX_UTIL}, POLL_SECONDS={POLL_SECONDS}",
        flush=True,
    )

    for mk in MODELS:
        pjson = INPUT_DIR / f"{mk}.all47.for_step5.with_gold.jsonl"
        if not pjson.exists():
            raise FileNotFoundError(
                f"Missing all47 Step5 prompt file: {pjson}\n"
                "Run scripts/build_all47_step5_prompts.py first."
            )

    queue = MODELS[:]
    running = {}
    failed = []

    while queue or running:
        done = []
        for mk, item in list(running.items()):
            code = item["proc"].poll()
            if code is not None:
                item["log_fh"].close()
                done.append(mk)
                if code == 0:
                    print(f"[done] {mk} on GPU {item['gpu']}", flush=True)
                else:
                    print(f"[fail] {mk} on GPU {item['gpu']} exit={code}", flush=True)
                    failed.append(mk)

        for mk in done:
            running.pop(mk, None)

        if failed:
            raise SystemExit(f"[FAIL] workers failed: {failed}")

        used_by_us = {v["gpu"] for v in running.values()}
        gpus = query_gpus()
        candidates = [
            g
            for g in gpus
            if g["idx"] not in used_by_us
            and g["free"] >= MIN_FREE_MIB
            and g["util"] <= MAX_UTIL
        ]

        while queue and candidates:
            mk = queue.pop(0)
            gpu = candidates.pop(0)["idx"]
            model_id = MODEL_ID[mk]
            subj_layer, obj_layer = get_layers(model_id)

            pjson = INPUT_DIR / f"{mk}.all47.for_step5.with_gold.jsonl"
            model_out = out / mk
            model_out.mkdir(parents=True, exist_ok=True)

            log_path = out / "logs" / f"{mk}.log"
            log_fh = open(log_path, "w", encoding="utf-8")

            cmd = [
                sys.executable,
                "scripts/lre_step5_deltacos_gold_per_triple.py",
                "--model_id",
                model_id,
                "--model_key",
                mk,
                "--prompts_jsonl",
                str(pjson),
                "--out_dir",
                str(model_out),
                "--train_frac",
                "0.75",
                "--seed",
                "0",
                "--batch_size",
                "8",
                "--subject_layer",
                str(subj_layer),
                "--object_layer",
                str(obj_layer),
            ]

            if USE_CHAT:
                cmd.append("--use_chat_template")

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

            print(
                f"[launch] {mk} -> GPU {gpu}, layers=({subj_layer},{obj_layer}), pid={proc.pid}, log={log_path}",
                flush=True,
            )

        if queue:
            gpu_state = ", ".join(
                [f"{g['idx']}:free={g['free']}MiB,util={g['util']}%" for g in gpus]
            )
            run_state = ", ".join([f"{mk}@{v['gpu']}" for mk, v in running.items()])
            print(
                f"[wait] queued={queue}; running={run_state}; gpus={gpu_state}",
                flush=True,
            )

        time.sleep(POLL_SECONDS)

    print(f"[done] OUT={out}", flush=True)


if __name__ == "__main__":
    main()
