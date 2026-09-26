"""Read/write the cached eval results shown by the Evals tab."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from triage.evals.runner import EvaluationRunner, sweep_retrieval_strategies
from triage.persist import load_records
from triage.rag.parse import IssueRecord

from webapp.repo_service import DATA_DIR, ensure_repo


def evals_path() -> Path:
    return DATA_DIR / "evals" / "evals.json"


def load_evals() -> dict | None:
    for path in (evals_path(), _seed_evals_path()):
        if path.exists():
            return json.loads(path.read_text())
    return None


def _seed_evals_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "seed" / "evals.json"


def precompute(repo: str, held_out_limit: int = 10) -> dict:
    """Build the eval repo if needed, run the comparison, and cache the results."""
    paths = ensure_repo(repo)
    runner = EvaluationRunner(paths.indexes, paths.processed)
    results = runner.run_comparison(limit=held_out_limit)
    sweep = sweep_retrieval_strategies(
        paths.processed, paths.indexes, held_out_limit=held_out_limit
    )
    payload = {
        "repo": repo,
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_size": len(load_records(paths.processed / "issues_corpus.json", IssueRecord)),
        "held_out_size": len(load_records(paths.processed / "issues_held_out.json", IssueRecord)),
        "systems": {
            name: {"system": name, **result.to_dict()}
            for name, result in results.items()
        },
        "retrieval_sweep": sweep,
    }
    _save(payload)
    return payload


def _save(payload: dict) -> None:
    path = evals_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
