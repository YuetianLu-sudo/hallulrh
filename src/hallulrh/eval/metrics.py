from __future__ import annotations

import csv
import math
from collections import defaultdict
from typing import Dict, List, Tuple


def wilson_ci(count: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Returns (low, high).
    """
    if n == 0:
        return (0.0, 0.0)
    p_hat = count / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (
        z
        * math.sqrt(
            (p_hat * (1 - p_hat) + z2 / (4 * n)) / n
        )
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_task_metrics(items: List[Dict]) -> Dict[str, Dict]:
    """
    Compute refusal / hallucination rates per task ("father", "instrument").

    Returns:
        {
          "father": {...},
          "instrument": {...},
        }
    """
    groups = defaultdict(list)
    for it in items:
        task = it.get("task", "unknown")
        groups[task].append(it)

    results: Dict[str, Dict] = {}

    for task, group in groups.items():
        n = len(group)
        n_refusal = sum(it.get("refusal", 0) for it in group)
        n_hallu = sum(it.get("hallucination", 0) for it in group)

        ref_rate = n_refusal / n if n > 0 else 0.0
        hallu_rate = n_hallu / n if n > 0 else 0.0

        ref_low, ref_high = wilson_ci(n_refusal, n)
        hal_low, hal_high = wilson_ci(n_hallu, n)

        results[task] = {
            "task": task,
            "n": n,
            "refusal_count": n_refusal,
            "refusal_rate": ref_rate,
            "refusal_ci_low": ref_low,
            "refusal_ci_high": ref_high,
            "hallucination_count": n_hallu,
            "hallucination_rate": hallu_rate,
            "hallucination_ci_low": hal_low,
            "hallucination_ci_high": hal_high,
        }

    return results


def write_metrics_csv(
    metrics: Dict[str, Dict],
    path: str,
) -> None:
    """
    Write metrics dict to a CSV file.
    """
    if not metrics:
        return

    fieldnames = [
        "task",
        "n",
        "refusal_count",
        "refusal_rate",
        "refusal_ci_low",
        "refusal_ci_high",
        "hallucination_count",
        "hallucination_rate",
        "hallucination_ci_low",
        "hallucination_ci_high",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for task in sorted(metrics.keys()):
            writer.writerow(metrics[task])

    print(f"[hallulrh] Metrics written to {path}")
