# Issue Triage Copilot

An AI copilot that recommends next actions for open-source issue triage. Given a new or
untriaged issue in a target repo, it recommends: suggested labels + triage route, concrete
next steps, likely affected modules, and 2-4 similar previously-resolved issues — all with
citations. Every recommendation is grounded in a cited source (issue ID, PR, or doc
section). The system recommends actions only; it never resolves issues or edits code.

## Architecture

Two RAG indexes are built at ingestion time from GitHub API data (no live API calls at
runtime):

- **Index A (issue history)** — resolved issues with title, body, author-tagged comments,
  and resolution metadata (applied labels, linked PRs, closing comments). The retrieval
  unit is the **whole issue**; the embedding signature is a normalized title+body (code
  fences and stack traces are kept), truncated to ~800 tokens. The full record is returned
  by issue ID.
- **Index B (process docs)** — CONTRIBUTING.md, docs, issue/PR templates, and release
  guides. Chunked with a markdown-heading-aware split (heading hierarchy as metadata) then
  a recursive token splitter (~600 tokens, 15% overlap). A flat chunker is also available
  for the chunking sweep.

### Multi-agent orchestration (LangGraph)

A single orchestrator agent drives the graph:

1. **plan** — classifies the issue (`classify_issue` MCP tool) and sets confidence.
2. **parallel specialists** — the historical-context agent (ReAct) gathers similar resolved
   issues and resolution patterns via `search_past_issues` + `get_issue_details`; the
   process agent (ReAct) gathers runbook steps via `get_runbook_steps`.
3. **decide** — re-enters to merge evidence, rank actions, call `get_resolution_patterns`,
   produce the final `TriageDecision` JSON, verify every citation, and gate on confidence.
   Low confidence routes to a **human-in-the-loop** edge.

### MCP tool layer

Five read-only tools over the local corpus, with strict per-agent access:

| Tool | Used by |
|---|---|
| `classify_issue` | Orchestrator |
| `get_resolution_patterns` | Orchestrator (ranking) |
| `search_past_issues` | Historical agent |
| `get_issue_details` | Historical agent |
| `get_runbook_steps` | Process agent |

No write tools exist. `guardrails/citation_verify.py` drops any cited issue ID or doc
section that does not exist before the decision reaches the final output.

### Vanilla RAG baseline

`agents/vanilla.py` + `rag/pipeline.py` retrieve from both indexes with one prompt (no
orchestration) and produce the same `TriageDecision` schema. The multi-agent uplift vs this
baseline is the key evaluation result.

## Folder layout

```
triage/
  data/raw, data/processed, data/indexes   # indexes + datasets are gitignored
  rag/        parse, chunk, embed, store, retriever, pipeline, index_build
  agents/     historical, process, vanilla, react
  orchestration/ state, graph, nodes, edges, human_in_loop
  mcp_tools/  server, tools, langchain (AgentToolbox + tool groups)
  memory/     store (evidence cache), session
  guardrails/ citation_verify, schema, confidence
  prompts/    orchestrator, historical, process, vanilla, judge
  evals/      dataset (held-out split), judge, metrics, runner
  scripts/    fetch, build_corpus, run_demo, run_evals
  tests/      per-module pytest files
```

## How to run

Requirements: Python 3.11+, a GitHub token for data fetching, and an OpenAI API key for
LLMs/embeddings.

```bash
pip install -e ".[dev]"
# put secrets in .env (gitignored) or the environment
GITHUB_TOKEN=...  OPENAI_API_KEY=...
```

Optional LLM model overrides: `OPENAI_MAIN_MODEL` (default `gpt-4o-mini`) and
`OPENAI_FAST_MODEL` (default `gpt-4o-mini`).

### 1. Fetch the dataset

```bash
python triage/scripts/fetch.py scikit-learn/scikit-learn pandas-dev/pandas --limit 1000
```

Fetches closed issues (comments, labels, linked PRs) and process docs, applies the held-out
split (15%), and persists `issues_corpus.json`, `issues_held_out.json`, `process_docs.json`
under `triage/data/processed/`.

### 2. Build the indexes

```bash
python triage/scripts/build_corpus.py
```

Embeds the corpus (excluding held-out) and writes Chroma indexes under
`triage/data/indexes/`. Chunking knobs: `--chunk-size`, `--overlap`, `--flat`.

### 3. Run the demo

```bash
python triage/scripts/run_demo.py --held-out-index 0
python triage/scripts/run_demo.py --text "Title\n\nbody of a brand-new issue"
```

Prints the final decision JSON, the confidence gate result, and (for held-out issues) the
actual resolution for comparison.

### 4. Run the evaluation

```bash
python triage/scripts/run_evals.py                # vanilla vs multi-agent table
python triage/scripts/run_evals.py --sweep        # + doc-chunking sweep
```

## Evaluation design

- **Held-out set** — 15-20% of resolved issues are carved out at ingestion time and never
  enter the corpus; they are treated as new issues at eval time.
- **LLM-as-judge** (`gpt-4o-mini`) on every held-out issue: answer relevancy, context
  relevance, groundedness.
- **Objective metrics** — label accuracy (top-1/top-3 vs actual applied labels), action
  overlap (ROUGE-L + entity match vs the actual closing comment / linked PR), and retrieval
  recall@k.
- **Chunking sweep** — structural vs flat × chunk size {300, 500, 700} × overlap {10%,
  15%}, reporting recall@k + context relevance to justify the chunking choice.
- **System metrics** — p95 latency and estimated cost per triage.

## Results

To be filled in after the first real run over the selected repos (see step 4). The template
table below is produced by `run_evals.py`:

```
system   label_top1  label_top3  action_rouge_l  action_entity_match  ...  latency_p95  cost
vanilla  ...         ...         ...             ...                 ...  ...          ...
multi    ...         ...         ...             ...                 ...  ...          ...
```

## Notes & scope

- Pipeline targets the chosen repos only; no generic multi-API support.
- No code indexing — file paths appear only as text inside issues.
- Secrets live only in `.env` / environment variables; never log or commit them.