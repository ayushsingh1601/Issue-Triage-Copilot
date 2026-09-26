#!/usr/bin/env python
"""Fetch resolved issues and process docs for the target repos and persist the dataset."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from triage.evals.dataset import held_out_split
from triage.fetch import fetch_issues, fetch_process_docs
from triage.github import GitHubClient
from triage.persist import save_records
from triage.rag.parse import IssueRecord, ProcessDoc, parse_docs, parse_issue


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repos", nargs="+", help="owner/repo pairs, e.g. scikit-learn/scikit-learn")
    parser.add_argument("--limit", type=int, default=None, help="max issues per repo")
    parser.add_argument("--out", type=Path, default=Path("triage/data/processed"))
    parser.add_argument("--held-out-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required (set it in .env or the environment)")

    records: list[IssueRecord] = []
    docs: list[ProcessDoc] = []
    with GitHubClient(token=token) as gh:
        for repo in args.repos:
            print(f"fetching {repo}...")
            fetched = fetch_issues(gh, repo, state="closed", limit=args.limit)
            records.extend(parse_issue(item) for item in fetched)
            docs.extend(parse_docs(fetch_process_docs(gh, repo)))
            print(f"  {len(records)} issues so far")

    split = held_out_split(records, held_out_fraction=args.held_out_fraction, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    save_records(split.corpus, args.out / "issues_corpus.json")
    save_records(split.held_out, args.out / "issues_held_out.json")
    save_records(docs, args.out / "process_docs.json")
    print(
        f"saved corpus={len(split.corpus)} held_out={len(split.held_out)} "
        f"docs={len(docs)} -> {args.out}"
    )


if __name__ == "__main__":
    main()
