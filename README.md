# Issue Triage Copilot

An AI copilot that recommends next actions for open-source issue triage. Given a new or
untriaged issue in a target repo, it recommends: suggested labels + triage route, concrete
next steps, likely affected modules, and 2-4 similar previously-resolved issues, all with
citations. Every recommendation is grounded in a cited source (issue ID, PR, or doc
section). The system recommends actions only; it never resolves issues or edits code.

## Architecture

- **Two RAG indexes** built at ingestion time from GitHub API data (no live API calls at
  runtime):
  - Index A (issue history): resolved issues with title, body, comments, and resolution
    metadata (applied labels, linked PRs, closing comments).
  - Index B (process docs): CONTRIBUTING.md, docs, issue/PR templates, release guides.
- **Multi-agent orchestration** via LangGraph with a single orchestrator agent: it
  classifies and plans, routes to two specialist agents in parallel (historical-context and
  process), then re-enters to merge evidence, rank actions, attach citations, and emit the
  final JSON decision. Low confidence routes to a human-in-the-loop edge.
- **Vanilla RAG baseline** (dual-index retrieval + one prompt, no orchestration) kept for
  comparison.
- **MCP tool layer**: 5 read-only tools over the local corpus with strict per-agent access.
- **Guardrails**: every cited issue ID / doc section is verified to exist before output.

## Repos

Target repos and dataset details are pinned down by a day-1 data audit. See
`triage/scripts/fetch.py`.

## How to run

```bash
pip install -e ".[dev]"
export GITHUB_TOKEN=...   # data fetch only
export OPENAI_API_KEY=... # runtime LLMs and embeddings
```

Steps are built incrementally (see git history). Run tests with `pytest`, lint with `ruff`.