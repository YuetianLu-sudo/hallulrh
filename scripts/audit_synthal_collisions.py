import argparse
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FIXED_15 = [
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

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"


def norm_text(x: str) -> str:
    x = unicodedata.normalize("NFKD", x)
    x = "".join(ch for ch in x if not unicodedata.combining(ch))
    x = x.casefold()
    x = re.sub(r"[\W_]+", "", x)
    return x.strip()


def sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "SyntHalCollisionAudit/1.0"})
    return s


def wikipedia_exact(session: requests.Session, subject: str):
    params = {
        "action": "query",
        "format": "json",
        "redirects": 1,
        "titles": subject,
    }
    r = session.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    out = []
    for _, page in pages.items():
        if "missing" not in page:
            out.append(page.get("title", ""))
    return out


def wikipedia_search(session: requests.Session, subject: str, limit: int = 5):
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": f"\"{subject}\"",
        "srlimit": limit,
    }
    r = session.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return [x.get("title", "") for x in data.get("query", {}).get("search", [])]


def wikidata_search(session: requests.Session, subject: str, limit: int = 10):
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "type": "item",
        "limit": limit,
        "search": subject,
    }
    r = session.get(WIKIDATA_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("search", [])


def audit_subject(subject: str, fuzzy_threshold: float):
    session = make_session()
    norm_subj = norm_text(subject)

    exact_hits = []
    best_source = ""
    best_candidate = ""
    best_score = 0.0
    best_desc = ""

    try:
        wiki_titles = wikipedia_exact(session, subject)
        for title in wiki_titles:
            score = sim(norm_subj, norm_text(title))
            if score > best_score:
                best_source, best_candidate, best_score = "wikipedia_exact", title, score
            if norm_text(title) == norm_subj:
                exact_hits.append(f"wikipedia_exact::{title}")

        wiki_search_titles = wikipedia_search(session, subject, limit=5)
        for title in wiki_search_titles:
            score = sim(norm_subj, norm_text(title))
            if score > best_score:
                best_source, best_candidate, best_score = "wikipedia_search", title, score
            if norm_text(title) == norm_subj:
                exact_hits.append(f"wikipedia_search::{title}")
    except Exception as e:
        wiki_search_titles = []
        wiki_titles = []
        best_desc = f"wiki_error::{type(e).__name__}"

    try:
        wd_hits = wikidata_search(session, subject, limit=10)
        for hit in wd_hits:
            label = hit.get("label", "") or ""
            desc = hit.get("description", "") or ""
            aliases = hit.get("aliases", []) or []
            candidates = [label] + aliases
            for cand in candidates:
                score = sim(norm_subj, norm_text(cand))
                if score > best_score:
                    best_source, best_candidate, best_score = "wikidata_search", cand, score
                    best_desc = desc
                if norm_text(cand) == norm_subj:
                    exact_hits.append(f"wikidata_search::{cand}::{desc}")
    except Exception as e:
        wd_hits = []
        if not best_desc:
            best_desc = f"wikidata_error::{type(e).__name__}"

    exact_hits = sorted(set(exact_hits))

    if exact_hits:
        verdict = "matched"
    elif best_score >= fuzzy_threshold:
        verdict = "ambiguous"
    else:
        verdict = "clean"

    return {
        "subject": subject,
        "norm_subject": norm_subj,
        "verdict": verdict,
        "exact_hits": " || ".join(exact_hits),
        "best_source": best_source,
        "best_candidate": best_candidate,
        "best_score": round(best_score, 6),
        "best_desc": best_desc,
        "wiki_exact_titles": " || ".join(wiki_titles if "wiki_titles" in locals() else []),
        "wiki_search_titles": " || ".join(wiki_search_titles if "wiki_search_titles" in locals() else []),
        "n_wikidata_hits": len(wd_hits) if "wd_hits" in locals() else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fuzzy-threshold", type=float, default=0.93)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cache_path = os.path.join(args.outdir, "subject_audit_cache.jsonl")

    rows = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("relation_key") in FIXED_15:
                rows.append(obj)

    prompts = pd.DataFrame(rows)
    prompts = prompts[["example_id", "relation_key", "relation_group", "subject", "template", "prompt"]].copy()

    subj_meta = (
        prompts.groupby("subject")
        .agg(
            n_prompts=("subject", "size"),
            relation_keys=("relation_key", lambda x: "|".join(sorted(set(x)))),
        )
        .reset_index()
    )

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cache[obj["subject"]] = obj

    todo = [s for s in subj_meta["subject"].tolist() if s not in cache]

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(audit_subject, s, args.fuzzy_threshold): s for s in todo}
            with open(cache_path, "a", encoding="utf-8") as out:
                for i, fut in enumerate(as_completed(futs), start=1):
                    obj = fut.result()
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    out.flush()
                    cache[obj["subject"]] = obj
                    if i % 10 == 0:
                        print(f"[audit] completed {i}/{len(todo)}")
                    time.sleep(0.02)

    audit = pd.DataFrame([cache[s] for s in subj_meta["subject"].tolist()])
    audit = subj_meta.merge(audit, on="subject", how="left")

    prompt_audit = prompts.merge(
        audit[["subject", "verdict", "exact_hits", "best_source", "best_candidate", "best_score", "best_desc"]],
        on="subject",
        how="left",
    )

    rel_summary = (
        prompt_audit.groupby(["relation_key", "verdict"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["clean", "ambiguous", "matched"]:
        if col not in rel_summary.columns:
            rel_summary[col] = 0

    rel_summary["total"] = rel_summary["clean"] + rel_summary["ambiguous"] + rel_summary["matched"]
    rel_summary["clean_rate"] = rel_summary["clean"] / rel_summary["total"]

    audit.to_csv(os.path.join(args.outdir, "subject_audit.csv"), index=False)
    prompt_audit.to_csv(os.path.join(args.outdir, "prompt_audit.csv"), index=False)
    rel_summary.to_csv(os.path.join(args.outdir, "relation_collision_summary.csv"), index=False)

    prompt_audit[prompt_audit["verdict"] == "clean"].to_json(
        os.path.join(args.outdir, "clean_prompts.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )
    prompt_audit[prompt_audit["verdict"] == "ambiguous"].to_json(
        os.path.join(args.outdir, "ambiguous_prompts.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )
    prompt_audit[prompt_audit["verdict"] == "matched"].to_json(
        os.path.join(args.outdir, "matched_prompts.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print("[write]", os.path.join(args.outdir, "subject_audit.csv"))
    print("[write]", os.path.join(args.outdir, "prompt_audit.csv"))
    print("[write]", os.path.join(args.outdir, "relation_collision_summary.csv"))
    print()
    print(rel_summary.sort_values(["clean_rate", "relation_key"]).to_string(index=False))


if __name__ == "__main__":
    main()
