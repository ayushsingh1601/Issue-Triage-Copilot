---
name: trace-runs
description: Set up, run, and verify tracing for triage runs — the local Tracer (per-node/LLM/tool latencies) and Langfuse (hosted traces of every agent call + the final decision). Use when debugging a run or checking observability.
---

# Trace runs (local Tracer + Langfuse)

Two tracing layers are available.

## 1. Local Tracer (no accounts)

`triage/tracing.py` records per-node, per-LLM and per-tool latencies. Enable with
`--trace <path>` on the demo:

```bash
python triage/scripts/run_demo.py --held-out-index 0 --trace triage/data/indexes/trace.json
```

`Tracer.summary()` gives count / avg / p95 per stage; `save()` persists the raw events.

## 2. Langfuse (hosted, free tier)

Set in `.env` or the environment (values may be wrapped in quotes; the loader strips them):

```bash
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or LANGFUSE_HOST / self-hosted URL
```

Requires `pip install -e ".[trace]"` (or `pip install langfuse`). When the two keys are
set, `triage/observability.py` attaches Langfuse's LangChain callback to every graph
invocation — the trace shows **each specialist LLM generation, each MCP tool call and its
output, and the final `TriageDecision` JSON**.

## Verify

1. `python -c "from triage.observability import langfuse_enabled; print(langfuse_enabled())"` → `True`.
2. Run the demo:
   ```bash
   python triage/scripts/run_demo.py --held-out-index 0
   ```
3. The run completes with no `Failed to export spans` errors; the Langfuse dashboard shows a
   trace with the agent calls and the decision.

## Troubleshooting

- Quoted values in `.env` (e.g. `KEY="value"`) are handled by the loader — do not include
  literal quotes.
- `Optional[RunnableConfig]` annotations are rewritten by ruff's `UP045`; LangGraph requires
  the node `config` param to be annotated `RunnableConfig` exactly.
- Notebooks run inside an event loop: use top-level `await` and `run_comparison_async`,
  never `asyncio.run()`.