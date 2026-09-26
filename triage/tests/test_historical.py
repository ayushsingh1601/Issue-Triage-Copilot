import asyncio
import json

from langchain_core.messages import AIMessage
from triage.agents.historical import HistoricalAgent
from triage.agents.process import ProcessAgent
from triage.mcp_tools.langchain import AgentToolbox
from triage.mcp_tools.tools import TriageTools
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


def make_tools(tmp_path) -> TriageTools:
    def fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    def fake_llm(prompt):
        return '{"type": "bug", "confidence": 0.9}'

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


class ScriptedModel:
    def __init__(self) -> None:
        self._steps = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_past_issues",
                        "args": {"query": "crash on empty frame", "limit": 3},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_issue_details",
                        "args": {"issue_id": "x/y#1"},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Similar crash in x/y#1, fixed by #45."),
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
                        "args": {"issue_type": "bug", "query": "crash on empty frame"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Runbook steps from CONTRIBUTING.md."),
        ]

    async def ainvoke(self, messages, config=None):
        return self._steps.pop(0)


def test_historical_agent_collects_evidence_and_citations(tmp_path):
    tools = make_tools(tmp_path)
    agent = HistoricalAgent(model=ScriptedModel())

    async def run():
        async with AgentToolbox(tools) as box:
            history_tools = await box.group("historical")
            return await agent.run("crash on empty frame", history_tools)

    evidence = asyncio.run(run())
    assert evidence["similar_issues"][0]["issue_id"] == "x/y#1"
    assert evidence["similar_issues"][0]["linked_prs"] == [45]
    assert evidence["citations"] == ["x/y#1", "x/y#2"]
    assert evidence["details"]
    assert evidence["details"][0]["comments"][0]["author_association"] == "MEMBER"
    assert "x/y#1" in evidence["summary"]


def orchestrator_json() -> str:
    return json.dumps(
        {
            "issue_id": "x/y#999",
            "suggested_labels": ["bug"],
            "triage_route": "bug",
            "next_steps": ["step"],
            "affected_modules": ["mod"],
            "similar_issues": [],
            "citations": ["x/y#1"],
        }
    )


def test_historical_node_in_graph(tmp_path):
    tools = make_tools(tmp_path)
    agent = HistoricalAgent(model=ScriptedModel())
    process_agent = ProcessAgent(model=ProcessScriptedModel())

    async def run():
        async with AgentToolbox(tools) as box:
            graph = build_graph(
                toolbox=box,
                historical_agent=agent,
                process_agent=process_agent,
                orchestrator_respond=lambda prompt: orchestrator_json(),
            ).compile()
            state = TriageState(
                issue="crash on empty frame",
                issue_id="x/y#999",
                classification={"type": "bug", "confidence": 0.9},
            )
            return await graph.ainvoke(state)

    result = asyncio.run(run())
    assert result["historical_evidence"]
    assert result["historical_evidence"][0]["issue_id"] == "x/y#1"
    assert "x/y#1" in result["citations"]
