---
name: setup-environment
description: Install the Issue Triage Copilot, configure .env secrets, and set optional model overrides. Use FIRST in a fresh clone, before any other triage skill.
---

# Set up the environment

Required: Python 3.11+, a GitHub token (data fetching), an OpenAI API key (LLMs and
embeddings).

## Install

```bash
pip install -e ".[dev]"        # add ",trace" to also install Langfuse (free tier)
```

## Secrets

Create `.env` (gitignored) or export in the environment:

```bash
GITHUB_TOKEN=...
OPENAI_API_KEY=...
```

Values may be wrapped in matching quotes (`KEY="value"`); the project's `.env` loader strips
them. The scripts `triage/scripts/fetch.py` and `triage/scripts/run_demo.py` load `.env`
automatically; `build_corpus.py` and `run_evals.py` read from the environment, so export
the vars (e.g. `set -a; . ./.env; set +a`) when running those.

## Optional overrides

- `OPENAI_MAIN_MODEL` (default `gpt-4o-mini`)
- `OPENAI_FAST_MODEL` (default `gpt-4o-mini`)
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` to enable Langfuse
  tracing (see the `trace-runs` skill)

## Verify

```bash
python -m pytest triage/tests
```

The suite is fully hermetic (no network or keys required).