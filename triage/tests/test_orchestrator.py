import asyncio
import json

from langchain_core.messages import AIMessage
from triage.agents.historical import HistoricalAgent
from triage.agents.process import ProcessAgent
from triage.guardrails.citation_verify import invalid_citations, verify_decision
from triage.guardrails.confidence import needs_human_review
from triage.guardrails.schema import TriageDecision
from triage.mcp_tools.langchain import AgentToolbox
from triage.mcp_tools.tools import TriageTools
from triage.memory.store import EvidenceStore
from triage.orchestration.graph import build_graph
from triage.orchestration.state import TriageState
from triage.persist import save_records
from triage.rag.parse import IssueComment, IssueRecord
from triage.rag.store import ChromaStore


def make_issue(repo: str, number: int) -> IssueRecord:
    return IssueRecord(
        repo=repo,
        number=number,
        title=f"Bug {number}",
        body="crash on empty frame with stack trace",
        state="closed",
        author="a",
        created_at="2024-01-01",
        closed_at="2024-01-02",
        labels=["bug"],
        linked_prs=[45],
        comments=[
            IssueComment(
                id=1,
                author="maint",
                author_association="MEMBER",
                body="fixed by #45",
                created_at="2024-01-02",
            )
        ],
    )


def make_tools(tmp_path, confidence: float = 0.9) -> TriageTools:
    def fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    def fake_llm(prompt):
        return json.dumps({"type": "bug", "confidence": confidence})

    records = [make_issue("x/y", 1), make_issue("x/y", 2)]
    save_records(records, tmp_path / "processed" / "issues_corpus.json")
    issue_store = ChromaStore(tmp_path / "indexes" / "issues", "issues")
    issue_store.add(
        ids=[f"{r.repo}#{r.number}" for r in records],
        texts=[f"{r.title}\n\n{r.body}" for r in records],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"repo": r.repo, "number": r.number} for r in records],
    )
    doc_store = ChromaStore(tmp_path / "indexes" / "docs", "docs")
    doc_store.add(
        ids=["CONTRIBUTING.md#top#0"],
        texts=["# Contributing\nReport bugs with a minimal repro."],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"repo": "x/y", "path": "CONTRIBUTING.md", "heading_path": ""}],
    )
    return TriageTools(
        indexes_dir=tmp_path / "indexes",
        processed_dir=tmp_path / "processed",
        embed_fn=fake_embed,
        llm_respond=fake_llm,
    )


class HistoricalScriptedModel:
    def __init__(self) -> None:
        self._steps = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_issue_details",
                        "args": {"issue_id": "x/y#1"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Similar crash in x/y#1."),
        ]

    async def ainvoke(self, messages, config=None):
        return self._steps.pop(0)


class ProcessScriptedModel:
    def __init__(self) -> None:
        self._steps = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_runbook_steps",
                        "args": {"issue_type": "bug", "query": "crash"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Runbook steps from CONTRIBUTING.md."),
        ]

    async def ainvoke(self, messages, config=None):
        return self._steps.pop(0)


def orchestrator_json() -> str:
    return json.dumps(
        {
            "issue_id": "x/y#999",
            "suggested_labels": ["bug", "DataFrame"],
            "triage_route": "bug: reproduce, add a regression test, route to core maintainers",
            "next_steps": ["Reproduce with a minimal example", "Add a failing test"],
            "affected_modules": ["pandas/core/frame.py"],
            "similar_issues": [
                {"issue_id": "x/y#1", "repo": "x/y", "title": "Bug 1", "reason": "same crash"},
                {"issue_id": "FAKE#1", "repo": "x/y", "title": "Fake", "reason": "hallucinated"},
            ],
            "citations": ["x/y#1", "FAKE#999", "CONTRIBUTING.md#top#0"],
        }
    )


def test_verify_decision_drops_unknown_sources():
    decision = TriageDecision.model_validate_json(orchestrator_json())
    issue_ids = {"x/y#1", "x/y#2"}
    doc_ids = {"CONTRIBUTING.md#top#0"}
    cleaned = verify_decision(decision, issue_ids=issue_ids, doc_ids=doc_ids)
    assert cleaned.citations == ["x/y#1", "CONTRIBUTING.md#top#0"]
    assert [s.issue_id for s in cleaned.similar_issues] == ["x/y#1"]
    assert invalid_citations(decision, issue_ids, doc_ids) == ["FAKE#999"]


def test_needs_human_review_gate():
    assert needs_human_review(0.9) is False
    assert needs_human_review(0.3) is True


def test_evidence_store_round_trip(tmp_path):
    store = EvidenceStore(path=tmp_path / "evidence.json")
    store.put("x/y#1", {"decision": {"labels": ["bug"]}})
    reloaded = EvidenceStore(path=tmp_path / "evidence.json")
    assert reloaded.get("x/y#1") == {"decision": {"labels": ["bug"]}}


def test_full_run_emits_final_json(tmp_path):
    tools = make_tools(tmp_path, confidence=0.9)
    store = EvidenceStore(path=tmp_path / "evidence.json")
    historical = HistoricalAgent(model=HistoricalScriptedModel())
    process = ProcessAgent(model=ProcessScriptedModel())

    async def run():
        async with AgentToolbox(tools) as box:
            graph = build_graph(
                toolbox=box,
                historical_agent=historical,
                process_agent=process,
                orchestrator_respond=lambda prompt: orchestrator_json(),
                evidence_store=store,
            ).compile()
            state = TriageState(issue="crash on empty frame", issue_id="x/y#999")
            return await graph.ainvoke(state)

    result = asyncio.run(run())
    decision = result["decision"]
    assert isinstance(decision, TriageDecision)
    assert decision.issue_id == "x/y#999"
    assert decision.suggested_labels == ["bug", "DataFrame"]
    assert decision.citations == ["x/y#1", "CONTRIBUTING.md#top#0"]
    assert [s.issue_id for s in decision.similar_issues] == ["x/y#1"]
    assert result["needs_human"] is False
    assert store.get("x/y#999")["decision"]["suggested_labels"] == ["bug", "DataFrame"]


def test_low_confidence_routes_to_human(tmp_path):
    tools = make_tools(tmp_path, confidence=0.3)
    historical = HistoricalAgent(model=HistoricalScriptedModel())
    process = ProcessAgent(model=ProcessScriptedModel())

    async def run():
        async with AgentToolbox(tools) as box:
            graph = build_graph(
                toolbox=box,
                historical_agent=historical,
                process_agent=process,
                orchestrator_respond=lambda prompt: orchestrator_json(),
            ).compile()
            state = TriageState(issue="unclear issue", issue_id="x/y#998")
            return await graph.ainvoke(state)

    result = asyncio.run(run())
    assert result["needs_human"] is True
    assert result["confidence"] == 0.3
