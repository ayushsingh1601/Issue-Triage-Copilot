"""Issue Triage Copilot — free hosted demo (Streamlit on HuggingFace Spaces).

Triage tab: enter any public GitHub repo + issue (number or pasted text) to
get a grounded multi-agent triage decision. Evals tab: cached comparison of
the vanilla-RAG baseline vs the multi-agent system (multi-agent is primary).
"""
from __future__ import annotations

import os
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from triage.logging import silence_libraries

from webapp.env import load_env_file
from webapp.evals_cache import load_evals, precompute
from webapp.repo_service import ensure_repo, find_issue, is_ready, validate_repo
from webapp.triage_service import summarize_issue, triage_issue

load_env_file()
silence_libraries()

for key in (
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "OPENAI_MAIN_MODEL",
    "OPENAI_FAST_MODEL",
    "EVAL_REPO",
    "MAX_ISSUES",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
):
    try:
        if key in st.secrets:
            os.environ.setdefault(key, str(st.secrets[key]))
    except Exception:
        break

st.set_page_config(page_title="Issue Triage Copilot", page_icon="\U0001f50d", layout="wide")

DEFAULT_REPO = "scikit-learn/scikit-learn"
EVAL_REPO = os.environ.get("EVAL_REPO", DEFAULT_REPO)

_SYSTEM_LABELS = {"vanilla": "Vanilla RAG (baseline)", "multi": "Multi-agent"}

_EVALS_WORKER_STARTED = False


def _valid_repo(repo: str) -> bool:
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", repo.strip()))


def render_triage() -> None:
    st.header("Triage an issue")
    st.caption(
        "Point at any public GitHub repo and an issue. The copilot grounds its decision in "
        "that repo's resolved issues and process docs."
    )

    repo = st.text_input("GitHub repo (owner/name)", value=DEFAULT_REPO)
    mode = st.radio("Issue", ["Issue number", "Paste issue text"], horizontal=True)
    number = 1
    text = ""
    if mode == "Issue number":
        number = st.number_input("Issue number", min_value=1, step=1, value=1)
    else:
        text = st.text_area("Issue title and body", height=180)
    run = st.button("Triage", type="primary", width="stretch")
    if not run:
        return

    if not _valid_repo(repo):
        st.error("Enter a repo as `owner/name`, e.g. `scikit-learn/scikit-learn`.")
        return
    if mode == "Paste issue text" and not text.strip():
        st.error("Paste some issue text.")
        return

    if not is_ready(repo):
        try:
            validate_repo(repo)
        except Exception as exc:
            st.error(f"Repo not usable: {exc}")
            return

    with st.status("Preparing repo (first triage fetches and indexes it)…") as status:

        def progress(message: str, pct: float | None) -> None:
            label = f"{message} {pct:.0%}" if pct is not None else message
            status.update(label=label)

        try:
            paths = ensure_repo(repo, progress=progress)
        except Exception as exc:
            status.update(label=f"Failed to prepare repo: {exc}", state="error")
            return
        status.update(label="Repo ready.", state="complete")

    if mode == "Issue number":
        record = find_issue(repo, number)
        if record is None:
            st.error(f"Could not find issue {repo}#{number}.")
            return
        query = f"{record.title}\n\n{record.body}"
        issue_id = f"{repo}#{number}"
        actual = summarize_issue(record)
    else:
        query = text.strip()
        issue_id = f"{repo}#fresh"
        actual = None

    with st.spinner("Triaging…"):
        result = triage_issue(paths, query, issue_id)

    _render_decision(result)
    if actual:
        _render_actual(actual)
        if result["decision"]:
            suggested = set(result["decision"]["suggested_labels"])
            actual_labels = set(actual["actual_labels"])
            st.caption(f"Suggested vs actual labels: {suggested or set()} vs {actual_labels}")


def _render_decision(result: dict) -> None:
    if result["rejected"]:
        st.warning(f"Query rejected by the input guard: {result['guard']['reason']}")
        return

    st.caption(
        f"Guard: {'allowed' if result['guard']['allowed'] else 'blocked'} — "
        f"{result['guard']['reason']}"
    )
    if result["needs_human"]:
        st.info("Low confidence — routed to human-in-the-loop review.")

    classification = result["classification"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Issue type", str(classification.get("type", "")))
    col2.metric("Confidence", f"{float(classification.get('confidence', 0.0)):.2f}")
    col3.metric("Latency", f"{result['latency_seconds']}s")

    decision = result["decision"]
    if not decision:
        st.warning("No decision produced.")
        return

    st.subheader("Suggested decision")
    st.markdown(f"**Triage route:** {decision['triage_route']}")
    labels = ", ".join(decision["suggested_labels"]) or "none"
    st.markdown(f"**Suggested labels:** {labels}")
    st.markdown("**Next steps:**")
    for step in decision["next_steps"]:
        st.markdown(f"- {step}")
    modules = ", ".join(decision["affected_modules"]) or "none"
    st.markdown(f"**Affected modules:** {modules}")
    st.markdown("**Similar resolved issues:**")
    if decision["similar_issues"]:
        for item in decision["similar_issues"]:
            st.markdown(f"- **{item['issue_id']}** — {item['title']} ({item['reason']})")
    else:
        st.markdown("none")
    citations = ", ".join(decision["citations"]) or "none"
    st.markdown(f"**Citations:** {citations}")

    with st.expander("Decision JSON"):
        st.json(decision)
    with st.expander("Trace (per-stage latency)"):
        if result["trace_summary"]:
            st.dataframe(result["trace_summary"], width="stretch")
        else:
            st.write("No trace events recorded.")


def _render_actual(actual: dict) -> None:
    st.subheader("Actual resolution")
    labels = ", ".join(actual["actual_labels"]) or "none"
    prs = ", ".join(f"#{p}" for p in actual["linked_prs"]) or "none"
    st.markdown(f"**State:** {actual['state']} · **Labels:** {labels} · **Linked PRs:** {prs}")
    if actual["closing_comment"]:
        st.markdown(f"**Closing comment:** {actual['closing_comment']}")


def render_evals() -> None:
    st.header("Evaluation")
    data = load_evals()
    if data is None:
        st.info(
            "No eval results cached yet. Precomputing the comparison on the demo repo "
            f"(`{EVAL_REPO}`) is a one-time background job that takes a few minutes."
        )
        if st.button("Run evals now"):
            with st.spinner("Running the comparison…"):
                precompute(EVAL_REPO)
            st.rerun()
        return

    systems = data["systems"]
    multi = systems.get("multi")
    vanilla = systems.get("vanilla")
    if not multi or not vanilla:
        st.warning("Cached evals are missing a system. Delete the cache and rerun.")
        return

    st.subheader("Multi-agent system — primary result")
    st.caption(
        f"On `{data['repo']}`: {data['corpus_size']} corpus issues, "
        f"{data['held_out_size']} held-out issues."
    )
    cols = st.columns(4)
    _metric(cols[0], "Label top-1", multi["label_top1"])
    _metric(cols[1], "Label top-3", multi["label_top3"])
    _metric(cols[2], "Groundedness", multi["groundedness"])
    _metric(cols[3], "Recall@10", multi["recall_at_k"])
    cols = st.columns(4)
    _metric(cols[0], "Precision@10", multi["precision_at_k"])
    _metric(cols[1], "Answer relevancy", multi["answer_relevancy"])
    _metric(cols[2], "p95 latency (s)", multi["latency_p95"])
    _metric(cols[3], "Cost / triage", multi["cost"], fmt=".6f")

    st.divider()
    st.subheader("Vanilla RAG vs Multi-agent")
    st.caption(
        "The multi-agent system above is the primary result; the baseline is shown for comparison."
    )
    rows = [
        {**result, "system": _SYSTEM_LABELS.get(name, name)}
        for name, result in systems.items()
    ]
    st.dataframe(rows, width="stretch")

    if data.get("retrieval_sweep"):
        st.subheader("Retrieval-strategy sweep")
        st.dataframe(data["retrieval_sweep"], width="stretch")

    st.caption(
        f"Generated {data['generated_at']}. Precomputed once and cached — refresh via "
        "`python webapp/precompute_evals.py`."
    )


def _metric(column, label: str, value: float, fmt: str = ".3f") -> None:
    column.metric(label, f"{value:{fmt}}")


def _start_evals_worker() -> None:
    global _EVALS_WORKER_STARTED
    if _EVALS_WORKER_STARTED or load_evals() is not None:
        return
    if not (os.environ.get("GITHUB_TOKEN") and os.environ.get("OPENAI_API_KEY")):
        return
    _EVALS_WORKER_STARTED = True
    threading.Thread(target=_run_precompute, daemon=True).start()


def _run_precompute() -> None:
    try:
        precompute(EVAL_REPO)
    except Exception as exc:
        print(f"evals precompute failed: {exc}")


def main() -> None:
    _start_evals_worker()
    tab_triage, tab_evals = st.tabs(["Triage", "Evals"])
    with tab_triage:
        render_triage()
    with tab_evals:
        render_evals()


if __name__ == "__main__":
    main()
