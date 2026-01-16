#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean / normalize a SyntHal release ZIP (schema + IDs).

What it does (deterministic):
  1) Remove the columns "source_file" and "task" from every JSONL record.
  2) Standardize entity_id to the majority convention:
        <task_key>_<0000-based task_idx>
     e.g., country_language_0000, father_0000, instrument_0000, ...
     (task_key is taken from the filename: SyntHal_<task_key>.jsonl)
  3) Update example_id accordingly: "<task_key>:<entity_id>"
  4) Run strict uniqueness checks (per task):
        - task_idx unique
        - entity_id unique (after rewrite)
        - subject unique
  5) Update SyntHal_manifest.json accordingly (fields + format_version + timestamp).

Usage (copy-paste):
  python clean_syntha_release_schema.py \
    --in_zip 1.zip \
    --out_zip SyntHal_release_clean.zip \
    --overwrite

This script only uses the Python standard library.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import zipfile
from typing import Dict, List, Optional, Tuple


CANONICAL_FIELDS: List[str] = [
    "dataset",
    "relation",
    "entity_type",
    "cohort",
    "entity_id",
    "subject",
    "prompt",
    "template",
    "task_idx",
    "example_id",
]


def _utc_now_z() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _detect_root_dir(zip_names: List[str]) -> str:
    """
    If the zip contains a single top-level directory (common for release zips),
    return it; else return "" (meaning "files are at zip root").
    """
    tops = set()
    for n in zip_names:
        if "/" in n:
            tops.add(n.split("/")[0])
    return next(iter(tops)) if len(tops) == 1 else ""


def _task_key_from_filename(root_dir: str, name: str) -> Optional[str]:
    """
    Match names like:
      <root_dir>/SyntHal_<task_key>.jsonl
    or (if root_dir==""):
      SyntHal_<task_key>.jsonl
    """
    if root_dir:
        pat = rf"^{re.escape(root_dir)}/SyntHal_(.+?)\.jsonl$"
    else:
        pat = r"^SyntHal_(.+?)\.jsonl$"
    m = re.match(pat, name)
    return m.group(1) if m else None


def _rewrite_record(rec: Dict, task_key: str, task_idx: int) -> Dict:
    """
    Drop unwanted keys and rewrite entity_id/example_id in a stable field order.
    """
    entity_id = f"{task_key}_{task_idx:04d}"

    out: Dict = {}

    # Keep (if present) in a stable order
    if "dataset" in rec:
        out["dataset"] = rec["dataset"]
    if "relation" in rec:
        out["relation"] = rec["relation"]
    if "entity_type" in rec:
        out["entity_type"] = rec["entity_type"]
    if "cohort" in rec:
        out["cohort"] = rec["cohort"]

    # Standardized identifiers
    out["entity_id"] = entity_id

    # Keep content
    if "subject" in rec:
        out["subject"] = rec["subject"]
    if "prompt" in rec:
        out["prompt"] = rec["prompt"]
    if "template" in rec:
        out["template"] = rec["template"]

    # Indices / ids
    out["task_idx"] = task_idx
    out["example_id"] = f"{task_key}:{entity_id}"

    return out


def clean_release_zip(in_zip: str, out_zip: str, overwrite: bool = False) -> None:
    if (not overwrite) and os.path.exists(out_zip):
        raise FileExistsError(f"Refusing to overwrite existing: {out_zip}")

    with zipfile.ZipFile(in_zip, "r") as zin:
        names = zin.namelist()
        root_dir = _detect_root_dir(names)

        # Discover all task jsonl files.
        task_files: List[Tuple[str, str]] = []
        for n in names:
            if not n.endswith(".jsonl"):
                continue
            if root_dir and not n.startswith(root_dir + "/"):
                continue
            task_key = _task_key_from_filename(root_dir, n)
            if task_key:
                task_files.append((task_key, n))

        if not task_files:
            raise ValueError(
                "No task JSONL files found. Expected files named like "
                "'SyntHal_<task>.jsonl' (optionally under a single root folder)."
            )

        manifest_name = f"{root_dir}/SyntHal_manifest.json" if root_dir else "SyntHal_manifest.json"
        has_manifest = manifest_name in names
        manifest: Optional[Dict] = json.loads(zin.read(manifest_name)) if has_manifest else None

        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            # Process and rewrite each task file.
            for task_key, fname in sorted(task_files, key=lambda x: x[0]):
                seen_subject = set()
                seen_task_idx = set()
                seen_entity_id = set()

                out_lines: List[str] = []

                with zin.open(fname) as f:
                    for line_no, raw in enumerate(f, start=1):
                        if not raw:
                            continue
                        rec = json.loads(raw.decode("utf-8"))

                        # Prefer existing task_idx if present; else use file order.
                        task_idx = int(rec.get("task_idx", line_no - 1))

                        subj = rec.get("subject", None)
                        if subj is not None:
                            if subj in seen_subject:
                                raise ValueError(f"[{task_key}] Duplicate subject: {subj!r}")
                            seen_subject.add(subj)

                        if task_idx in seen_task_idx:
                            raise ValueError(f"[{task_key}] Duplicate task_idx: {task_idx}")
                        seen_task_idx.add(task_idx)

                        entity_id = f"{task_key}_{task_idx:04d}"
                        if entity_id in seen_entity_id:
                            raise ValueError(f"[{task_key}] Duplicate entity_id after rewrite: {entity_id}")
                        seen_entity_id.add(entity_id)

                        rec_out = _rewrite_record(rec, task_key=task_key, task_idx=task_idx)
                        out_lines.append(json.dumps(rec_out, ensure_ascii=False))

                out_name = f"{root_dir}/SyntHal_{task_key}.jsonl" if root_dir else f"SyntHal_{task_key}.jsonl"
                zout.writestr(out_name, "\n".join(out_lines) + "\n")

            # Rewrite / create manifest.
            if manifest is None:
                # Minimal manifest if none existed
                manifest = {
                    "dataset": "SyntHal",
                    "format_version": "v1-clean",
                    "n_total": None,
                    "tasks": {k: None for k, _ in task_files},
                    "fields": CANONICAL_FIELDS,
                    "files": [f"SyntHal_{k}.jsonl" for k, _ in sorted(task_files)],
                }

            manifest_out = dict(manifest)
            fv = str(manifest_out.get("format_version", "v1"))
            if not fv.endswith("-clean"):
                manifest_out["format_version"] = fv + "-clean"
            manifest_out["fields"] = CANONICAL_FIELDS
            manifest_out["generated_at_utc"] = _utc_now_z()

            zout.writestr(manifest_name, json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n")

            # Copy any other files (excluding original task JSONLs + original manifest).
            for n in names:
                if n == manifest_name:
                    continue
                if n.endswith(".jsonl") and _task_key_from_filename(root_dir, n) is not None:
                    continue
                # Keep only files under root_dir if root_dir exists.
                if root_dir and not n.startswith(root_dir + "/"):
                    continue
                data = zin.read(n)
                zout.writestr(n, data)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean SyntHal release zip schema (drop task/source_file; standardize entity_id)."
    )
    p.add_argument("--in_zip", required=True, help="Input release zip (e.g., 1.zip)")
    p.add_argument("--out_zip", required=True, help="Output cleaned zip (e.g., SyntHal_release_clean.zip)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite out_zip if it exists.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    clean_release_zip(args.in_zip, args.out_zip, overwrite=bool(args.overwrite))
    print(f"[ok] wrote: {args.out_zip}")


if __name__ == "__main__":
    main()
