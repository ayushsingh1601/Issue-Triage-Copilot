---
name: run-evaluation
description: Run the vanilla-RAG vs multi-agent comparison and the doc-chunking / retrieval-strategy sweeps over held-out issues. Use to measure system quality and justify choices.
---

# Run the evaluation

Runs `triage/scripts/run_evals.py`: vanilla baseline vs multi-agent over the held-out set,
with label accuracy, action overlap, binary LLM-judge scores, recall@k + precision@k, p95
latency, and estimated cost.

## Requirements

- `OPENAI_API_KEY` in the environment (`run_evals.py` does not load `.env`; export it,
  e.g. `set -a; . ./.env; set +a`).
- Built indexes (`build-indexes` skill).

## Command

```bash
# vanilla vs multi-agent table over the first N held-out issues
python triage/scripts/run_evals.py --held-out-limit N

# doc-chunking sweep (structural vs flat x {300,500,700} x {10%,15%})
python triage/scripts/run_evals.py --sweep --held-out-limit N

# retrieval-strategy sweep (base / rewrite / rerank / rewrite+rerank)
python triage/scripts/run_evals.py --sweep-retrieval --held-out-limit N
```

## Output

- A `system` table (vanilla vs multi) and, when requested, sweep tables.
- Judge scores are per-metric binary `yes`/`no` verdicts (one LLM call per metric).

## Notes

- Each held-out issue triggers several LLM calls; keep `--held-out-limit` small for a quick
  sanity check.
- The runner also exposes async APIs (`run_comparison_async`, `evaluate_system_async`) for
  use inside the notebook, which runs in an event loop.
- The multi-agent path runs in a single event loop; a per-issue `asyncio.run` loop would
  break with reused async clients (fixed in git history).

## Verify

The table prints both `vanilla` and `multi` rows with populated numeric columns.