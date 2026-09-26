---
name: run-triage
description: Run the multi-agent issue-triage demo on a held-out or fresh issue and inspect the decision JSON against the actual resolution. Use to inspect a single triage end-to-end.
---

# Run the triage demo

Runs `triage/scripts/run_demo.py` (multi-agent graph: input guard → classify → parallel
historical + process agents → decide with citation verification and a confidence gate).

## Requirements

- `OPENAI_API_KEY` in `.env` or the environment (`run_demo.py` loads `.env`).
- Built indexes (`build-indexes` skill).

## Command

```bash
# triage the Nth held-out issue (compares against the actual resolution)
python triage/scripts/run_demo.py --held-out-index 0

# triage a brand-new issue
python triage/scripts/run_demo.py --text "Title\n\nbody of the issue"
```

## Flags

- `--no-guard` disable the input guard (injection + relevance gate)
- `--no-rewrite` / `--no-rerank` disable the retrieval rewriter / LLM reranker
- `--trace <path>` save a per-stage latency trace

## Output

Guard result, classification, the final decision as a **readable summary plus the JSON**
(`format_decision`: labels, triage route, next steps, affected modules, similar issues +
citations), `needs_human`, and — for held-out issues — the actual labels / linked PRs /
closing comment.

## Notes

- If the decision has no `similar_issues`/`citations`, the corpus is likely too small for
  the retrieval step to surface relevant evidence.
- Add `--trace <path>` to capture per-stage latency (see the `trace-runs` skill).
- With no `--text` and no `--held-out-index`, the script exits with a usage error.