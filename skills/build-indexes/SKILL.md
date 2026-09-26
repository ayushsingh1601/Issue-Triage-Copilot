---
name: build-indexes
description: Build or incrementally refresh the two RAG indexes (issue history + process docs) in ChromaDB. Use after fetching the dataset, after new issues arrive, or when TriageTools warns the index is stale.
---

# Build / refresh the RAG indexes

Runs `triage/scripts/build_corpus.py`. Default is a clean build (collections cleared first,
embedding cache kept). `--refresh` incrementally adds newly-ingested issues and deletes
evicted ones without re-embedding unchanged texts.

## Requirements

- `OPENAI_API_KEY` in the environment (`build_corpus.py` does not load `.env`; export it,
  e.g. `set -a; . ./.env; set +a`).

## Command

```bash
# clean build
python triage/scripts/build_corpus.py

# incremental refresh after new issues were fetched
python triage/scripts/build_corpus.py --refresh
```

## Output

- Chroma collections `issues` and `docs` under `triage/data/indexes/`
- Embedding cache `triage/data/indexes/embeddings.json`

## Notes

- The RAG DB is built once at ingestion and reused for every query; it is never rebuilt per
  issue.
- A stale index surfaces as a startup warning from `TriageTools` (see `index_stats()`);
  run `--refresh` to fix.

## Verify

Clean build prints `built issue index (N issues) and doc index (M docs)`; refresh prints
`added=A removed=R unchanged=U`.