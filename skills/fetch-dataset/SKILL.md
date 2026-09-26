---
name: fetch-dataset
description: Fetch closed GitHub issues and process docs for the target repos, apply the held-out split, and persist the dataset. Use when starting from scratch or re-fetching to pick up new issues.
---

# Fetch the dataset

Runs `triage/scripts/fetch.py`: pulls closed issues (comments, labels, linked PRs via issue
events) plus process docs (CONTRIBUTING, docs, templates, release guides), applies the 15%
held-out split (seed 42), and persists the corpus, held-out set, and docs.

## Requirements

- `GITHUB_TOKEN` in `.env` or the environment.

## Command

```bash
python triage/scripts/fetch.py <owner/repo>... --limit N
```

Example:

```bash
python triage/scripts/fetch.py scikit-learn/scikit-learn --limit 3000
```

## Output

- `triage/data/processed/issues_corpus.json`
- `triage/data/processed/issues_held_out.json`
- `triage/data/processed/process_docs.json`

## Notes

- Re-running overwrites the corpus; issues beyond `--limit` are evicted.
- After fetching, build or refresh the indexes with the `build-indexes` skill.

## Verify

Check the printed counts, e.g. `corpus=2550 held_out=450 docs=6`.