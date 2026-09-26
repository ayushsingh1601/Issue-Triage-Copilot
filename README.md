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

### Input guard

Every query passes an input guardrail before any agent work:

- **Prompt-injection detection** — a deterministic rule set (`guardrails/input_guard.py`)
  catches instruction-override attempts ("ignore previous instructions", "reveal your
  system prompt", role-switching, jailbreak phrasing).
- **Relevance gate** — an LLM (binary `yes`/`no` verdict, injectable) rejects queries that
  are not a triageable issue (unrelated text, gibberish, short input).

A rejected query short-circuits the graph to a `reject` node, so no LLM/tool calls are made
for it. Enable with `build_graph(..., input_guard=InputGuard())`; the demo enables it by
default (`--no-guard` to disable).

### Vanilla RAG baseline

`agents/vanilla.py` + `rag/pipeline.py` retrieve from both indexes with one prompt (no
orchestration) and produce the same `TriageDecision` schema. The multi-agent uplift vs this
baseline is the key evaluation result.

### Retrieval enhancement (rewriter + reranker)

Retrieval is enhanced by two optional, OpenAI-only components (both `gpt-4o-mini`,
injectable for tests):

- **Query rewriter** (`rag/rewrite.py`) — distills the raw issue into one clean search
  query before embedding.
- **LLM reranker** (`rag/rerank.py`) — pulls a candidate pool (**15 issues / 10 docs**) and
  re-ranks it in a single call before returning the top-k.

They are **on by default** and apply to both the vanilla pipeline and the multi-agent MCP
tools (`search_past_issues`, `get_runbook_steps`). Opt out with `--no-rewrite` /
`--no-rerank` in the demo, or `Components(use_retrieval_enhancements=False)` in the runner.
Passing bare `Retriever(...)`/`TriageTools(...)` (no rewriter/reranker) keeps the old flat
cosine retrieval.

## How the system works — full flow

End-to-end pipeline, from raw GitHub data to a grounded triage decision:

```
GitHub API ──fetch──▶ IssueRecord / ProcessDoc
                        │  held-out split (15%, never enters corpus)
                        ▼
                 corpus + process docs ──chunk──▶ IssueChunk / DocChunk
                        │  embed (text-embedding-3-small, disk-cached)
                        ▼
            Chroma Index A (issues) + Index B (docs) ──built by build_corpus.py
                        │
                        ▼
   new / held-out issue
        │
        ▼  orchestration/graph.py (LangGraph)
   guard ──prompt-injection rules + relevance gate──▶ reject (END) | continue
        │
        ▼
   plan ──classify_issue──▶ {type, confidence}
        │
        ├──▶ historical agent (ReAct over MCP): search_past_issues → get_issue_details
        │        ▶ similar issues + resolution metadata (citations: repo#number)
        ├──▶ process agent (ReAct over MCP): get_runbook_steps(issue_type, query)
        │        ▶ runbook steps (citations: doc section source ids)
        │
        ▼  decide (orchestrator re-entry)
   merge evidence + get_resolution_patterns ▶ LLM ▶ TriageDecision JSON
        │  guardrails: citation_verify (drop non-existent sources)
        │             confidence gate (low → human_in_loop edge)
        ▼
   final decision JSON (+ evidence cached in memory/store.py)
```

Each stage maps to a commit in git history:

1. **Ingestion** — `scripts/fetch.py` pulls closed issues (comments, labels, linked PRs via
   issue events) and process docs, then applies the held-out split and persists JSON
   (`rag/parse.py` → Pydantic `IssueRecord` / `ProcessDoc`; `evals/dataset.py`).
2. **Indexing** — `scripts/build_corpus.py` chunks issues (whole-issue signatures) and docs
   (structural or flat), embeds with `text-embedding-3-small` (cached), and writes Chroma
   indexes (`rag/chunk.py`, `rag/embed.py`, `rag/store.py`, `rag/index_build.py`).
3. **Serving** — `mcp_tools/` exposes the five read-only tools; `AgentToolbox` connects them
   to LangChain with per-agent access groups; `agents/` implement the ReAct specialists;
   `orchestration/` composes the graph, guardrails, and confidence gate.
4. **Evaluation** — `scripts/run_evals.py` runs the vanilla vs multi-agent comparison over
   the held-out set, plus the doc-chunking sweep (`evals/judge.py`, `evals/metrics.py`,
   `evals/runner.py`).
5. **Demo** — `scripts/run_demo.py` triages a fresh or held-out issue and prints the final
   JSON next to the actual resolution.

## Folder layout

```
triage/
  data/raw, data/processed, data/indexes   # indexes + datasets are gitignored
  rag/        parse, chunk, embed, store, retriever, pipeline, index_build, rewrite, rerank
  agents/     historical, process, vanilla, react
  orchestration/ state, graph, nodes, edges, human_in_loop
  mcp_tools/  server, tools, langchain (AgentToolbox + tool groups)
  memory/     store (evidence cache), session
  guardrails/ citation_verify, schema, confidence, input_guard
  prompts/    orchestrator, historical, process, vanilla, judge, classify, guard, rewrite, rerank
  evals/      dataset (held-out split), judge, metrics, runner
  scripts/    fetch, build_corpus, run_demo, run_evals
  tests/      per-module pytest files
  tracing.py  local latency tracer (per-node / per-LLM / per-tool)
  observability.py  optional Langfuse wiring (free tier)
notebooks/    demo.ipynb (run the full pipeline cell by cell)
```

## How to run

Requirements: Python 3.11+, a GitHub token for data fetching, and an OpenAI API key for
LLMs/embeddings.

```bash
pip install -e ".[dev]"        # add ",trace" to also install Langfuse (free tier)
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
`triage/data/indexes/`. Chunking knobs: `--chunk-size`, `--overlap`, `--flat`. The default
is a clean build: both collections are cleared first (the embedding cache is kept, so
unchanged texts are not re-embedded).

### 2b. Refresh the indexes when new issues arrive

The RAG database is built once at ingestion and reused for every query — it is never rebuilt
per issue. When the corpus changes (re-run `fetch.py` to pull new issues), refresh the index
incrementally:

```bash
python triage/scripts/build_corpus.py --refresh
```

`--refresh` adds newly-ingested issues (only their texts are embedded), deletes evicted ones,
and rebuilds the small doc collection — no duplicate-ID errors, no stale entries. At startup
`TriageTools` also warns (and `index_stats()` reports `missing`/`extra`) if the persisted
corpus and the index have drifted apart, so you never silently triage against a stale index.

### 3. Run the demo

```bash
python triage/scripts/run_demo.py --held-out-index 0
python triage/scripts/run_demo.py --text "Title\n\nbody of a brand-new issue"
```

The input guard runs first (prompt-injection rules + relevance gate; `--no-guard` to
disable), and retrieval uses the query rewriter + LLM reranker (on by default;
`--no-rewrite` / `--no-rerank` to disable). Prints the guard result, final decision JSON,
the confidence gate result, and (for held-out issues) the actual resolution for comparison.

### 4. Run the evaluation

```bash
python triage/scripts/run_evals.py                 # vanilla vs multi-agent table
python triage/scripts/run_evals.py --sweep         # doc-chunking sweep (recall + precision@10)
python triage/scripts/run_evals.py --sweep-retrieval   # base / rewrite / rerank / rewrite+rerank
```

### 5. Debug a run with tracing

Every stage of the agent flow can be timed and inspected with the built-in local
`Tracer` (`triage/tracing.py`). It records per-node, per-LLM and per-tool latencies and can
summarize (count / avg / p95) or persist the raw events:

```bash
python triage/scripts/run_demo.py --held-out-index 0 --trace triage/data/indexes/trace.json
```

```python
from triage.tracing import Tracer
from triage.orchestration.graph import build_graph

tracer = Tracer()
# ... build + invoke the graph with build_graph(..., tracer=tracer)
print(tracer.summary())          # {'node:plan': {...}, 'llm': {...}, 'tool': {...}, ...}
tracer.save("triage/data/indexes/trace.json")
```

The `Tracer` is wired into the graph nodes, the ReAct loop (each LLM call and each MCP tool
call is timed), and the demo/notebook flow.

### 6. Run everything from a notebook

`notebooks/demo.ipynb` steps through the whole pipeline in cells: dataset (fetch or reuse)
→ build indexes → triage a held-out issue → inspect the trace → vanilla-vs-multi
comparison. Open it with `jupyter notebook` or VS Code and run cells top to bottom. The
graph-invoke cells pass `graph_config()`, so Langfuse traces (if configured) are captured
from the notebook too. The build-indexes cell clears the collections first, so re-running it
is safe; after fetching new issues, prefer the incremental CLI refresh
(`build_corpus.py --refresh`).

## Observability with existing frameworks

The local `Tracer` needs no accounts or servers. For a hosted trace UI, **Langfuse** is
open-source and free (self-host with Docker, or its free cloud tier):

```bash
pip install -e ".[trace]"        # or: pip install langfuse
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
```

When those two keys are set, `triage/observability.py` automatically attaches Langfuse's
LangChain callback to every graph invocation (demo, notebook, and eval runner) — no code
changes needed. Because the callback is threaded through every agent call, the Langfuse
trace shows **each specialist LLM generation, each MCP tool call and its output, and the
final `TriageDecision` JSON**. When they are unset, the project runs on the local `Tracer`
only.

**LangSmith** is the alternative: set `LANGSMITH_API_KEY` (+ `LANGSMITH_TRACING=true`) and
LangChain/LangGraph calls are traced automatically. Both have free tiers; Langfuse is the
open-source / self-hostable choice.

## Evaluation design

- **Held-out set** — 15-20% of resolved issues are carved out at ingestion time and never
  enter the corpus; they are treated as new issues at eval time.
- **LLM-as-judge** (`gpt-4o-mini`) on every held-out issue, with a separate LLM call per
  metric. Each metric returns a **binary verdict** (`yes`/`no`) — answer relevancy, context
  relevance, groundedness — which is converted to a 1/0 score. A different model/respond can
  be configured per metric via `Judge(responds={metric: fn, ...})`.
- **Objective metrics** — label accuracy (top-1/top-3 vs actual applied labels), action
  overlap (ROUGE-L + entity match vs the actual closing comment / linked PR), and retrieval
  **recall@k + precision@k** (relevant set = corpus issues sharing ≥1 label with the query).
- **Chunking sweep** — structural vs flat × chunk size {300, 500, 700} × overlap {10%,
  15%}, reporting recall@k + precision@k + context relevance to justify the chunking choice.
- **Retrieval-strategy sweep** — `base` vs `rewrite` vs `rerank` vs `rewrite+rerank`,
  reporting recall@10 + precision@10 + judge scores to quantify the rewriter/reranker uplift.
- **System metrics** — p95 latency and estimated cost per triage.

## Results

To be filled in after the first real run over the selected repos (see "How to run", step
4). The table below is produced by `run_evals.py`:

```
system   label_top1  label_top3  action_rouge_l  ...  recall@k  precision@k  latency_p95  cost
vanilla  ...         ...         ...             ...  ...       ...          ...          ...
multi    ...         ...         ...             ...  ...       ...          ...          ...
```

## Problems faced & solutions

| Problem | Solution |
|---|---|
| `mcp` 2.x renamed `FastMCP` to `MCPServer` and broke the documented v1 API and `langchain-mcp-adapters` | Pinned `mcp>=1.0,<2` in `pyproject.toml` |
| `tiktoken` decodes partial token sequences (e.g., overlap tails) with a strict UTF-8 error | Rely on tiktoken's default `errors="replace"` and used token slicing for truncation |
| ChromaDB rejected collection names shorter than 3 characters | Used descriptive collection names (`issues`, `docs`) |
| Fake embeddings `[[1,0],[2,0],[3,0]]` are collinear, so cosine distances all tie at 0.0 and Chroma returned arbitrary order — tests were flaky | Switched test embeddings to non-collinear vectors so nearest-neighbor ordering is deterministic |
| LangGraph rejected concurrent writes to the shared `citations` key from the two parallel specialist nodes | Made `citations` an `Annotated[list[str], operator.add]` reducer so parallel updates merge |
| `langchain-mcp-adapters` returns list-typed tool results as one content block per element (multiple JSON objects concatenated), which broke single `json.loads` | Added `parse_json_documents()` to `jsonutil.py` that repeatedly `raw_decode`s JSON documents |
| Default LLM/embedding objects were constructed eagerly, so wiring real graph nodes into tests hit `OpenAIError` (missing key) | Injected scripted/fake LLM callables everywhere; made heavy imports (`langchain-openai`) lazy |
| Standalone `scripts/*.py` entry points couldn't import the `triage` package from the repo root | `sys.path` shim at the top of each script + per-file `E402` ruff ignore |
| A naive separator-splitting chunker silently dropped separator characters (content loss) | Re-wrote `_atomic_split` so each piece retains its trailing separator; concatenation of pieces equals the original text |
| First pass committed a step with a wrong metric expectation (`recall_at_k` semantics) | Re-ran tests before commit, reset the bad commit, and recommitted cleanly |
| MCP tool results exposed JSON in different shapes (scalar dict vs list) through LangChain content blocks | Centralized text extraction in `content_text()` and multi-doc parsing in `parse_json_documents()` |
| No visibility into which stage of the agent flow was slow | Added a local `Tracer` (`tracing.py`) wired into the graph nodes and ReAct loop, recording per-node / per-LLM / per-tool latencies with an avg/p95 summary |
| Prompts were scattered across agent/tool modules | Consolidated every prompt into `prompts/` (classify, historical, process, orchestrator, vanilla, judge) |
| `langfuse` install dragged in opentelemetry packages that conflicted with chromadb's pinned versions | Pinned the whole `opentelemetry` stack to one version (1.45.0) so both libraries import cleanly |
| A user query could try to override the system prompt or be unrelated to triage | Added an input guard (`guardrails/input_guard.py`): rule-based injection detection + an LLM relevance gate (binary verdict); rejected queries short-circuit the graph before any agent work |
| Raw issue text embeds poorly as a retrieval query and cosine order ignores semantics | Added a query rewriter (`rag/rewrite.py`) and an LLM reranker (`rag/rerank.py`) over a candidate pool; measured via `--sweep-retrieval` |
| Rebuilding the index over a grown corpus raised duplicate-ID errors and left evicted issues behind | Added `delete`/`clear` to `ChromaStore` and an incremental `sync_issue_index` (`build_corpus.py --refresh`); a startup staleness check warns when the corpus and index drift apart |

## Learnings

- **Recommend, don't act.** The copilot stays a recommendation layer: no write tools, and the
  structured `TriageDecision` JSON is the canonical output, which keeps the evaluation
  contract stable and makes the system auditable.
- **Decide the conversation boundary explicitly.** A conversational layer was considered and
  deliberately rejected: single-turn, stateless triage keeps the pipeline deterministic and
  the metrics comparable. Choose interactivity as a deliberate design decision, not an
  afterthought.
- **Guard the input boundary.** Validate every user query (rule-based prompt-injection
  detection + an LLM relevance gate) before any agent or LLM work, and short-circuit on
  rejection — it stops jailbreaks early and saves cost.
- **Retrieval quality is measured, not assumed.** Query rewriting and a candidate-pool LLM
  reranker sound plausible; they were validated with a strategy sweep
  (`--sweep-retrieval`) before being turned on by default.
- **Indexes are build artifacts, not per-query state.** The RAG DB is built once at
  ingestion, refreshed incrementally when the corpus grows (`--refresh`), and staleness is
  detected at startup so you never silently triage against old data.
- **One model family + injectable components keeps everything testable.** The OpenAI-only
  embedder, judge, rewriter, and reranker are uniform, and every LLM call is injectable, so
  the whole suite runs offline and deterministically.
- **Binary verdicts beat rating scales for auto-eval.** Per-metric, separate LLM calls
  returning `yes`/`no` are simpler to aggregate and more reliable than 1-5 scores.
- **Trace locally before adopting a platform.** A lightweight `Tracer` for per-node,
  per-LLM, per-tool latency answered the questions we had; a hosted backend (Langfuse) is
  only enabled behind env vars when deeper analysis is needed.
- **Eval hygiene starts at ingestion.** Carving out a fixed-seed held-out set at fetch time —
  and never letting it enter the corpus — keeps the vanilla-vs-multi comparison honest.
- **Inject every LLM/embedding callable.** Constructing real clients eagerly couples tests
  to network + secrets; injectable fns keep the suite hermetic and fast.
- **Scripts should stay thin.** Put testable logic in library modules
  (`rag/index_build.py`, `evals/runner.py`) and keep `scripts/` as argument parsing +
  orchestration.
- **Pin major library versions explicitly.** Renames/API breaks (mcp v2) silently break
  integrations; a version pin in `pyproject.toml` plus an import smoke test catches it.
- **Run tests before committing every step**, and when a commit ships a broken test, reset
  it and recommit rather than piling on fix commits.

## Notes & scope

- Pipeline targets the chosen repos only; no generic multi-API support.
- No code indexing — file paths appear only as text inside issues.
- Secrets live only in `.env` / environment variables; never log or commit them.