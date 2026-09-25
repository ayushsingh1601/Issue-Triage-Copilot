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
from triage.rag.parse import IssueRecord
from triage.rag.store import ChromaStore


def make_tools(tmp_path) -> TriageTools:
    def fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    def fake_llm(prompt):
        return '{"type": "bug", "confidence": 0.9}'

    record = IssueRecord(
        repo="x/y",
        number=1,
        title="Bug 1",
        body="crash on empty frame",
        state="closed",
        author="a",
        created_at="2024-01-01",
        closed_at="2024-01-02",
        labels=["bug"],
        linked_prs=[],
        comments=[],
    )
    save_records([record], tmp_path / "processed" / "issues_corpus.json")
    issue_store = ChromaStore(tmp_path / "indexes" / "issues", "issues")
    issue_store.add(
        ids=["x/y#1"],
        texts=["Bug 1\n\ncrash on empty frame"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"repo": "x/y", "number": 1}],
    )
    doc_store = ChromaStore(tmp_path / "indexes" / "docs", "docs")
    doc_store.add(
        ids=["CONTRIBUTING.md#Intro#0"],
        texts=["# Intro\nReport bugs with a minimal repro and full stack trace."],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"repo": "x/y", "path": "CONTRIBUTING.md", "heading_path": "Intro"}],
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
                        "name": "get_runbook_steps",
                        "args": {"issue_type": "bug", "query": "crash on empty frame"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Follow the bug-report runbook from CONTRIBUTING.md."),
        ]

    async def ainvoke(self, messages):
        return self._steps.pop(0)


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
            AIMessage(content="Reviewed x/y#1."),
        ]

    async def ainvoke(self, messages):
        return self._steps.pop(0)


def test_process_agent_returns_runbook_steps(tmp_path):
    tools = make_tools(tmp_path)
    agent = ProcessAgent(model=ScriptedModel())

    async def run():
        async with AgentToolbox(tools) as box:
            process_tools = await box.group("process")
            return await agent.run("bug", "crash on empty frame", process_tools)

    evidence = asyncio.run(run())
    assert evidence["steps"]
    step = evidence["steps"][0]
    assert step["path"] == "CONTRIBUTING.md"
    assert step["heading"] == "Intro"
    assert step["source"] == "CONTRIBUTING.md#Intro#0"
    assert evidence["citations"] == ["CONTRIBUTING.md#Intro#0"]
    assert "CONTRIBUTING.md" in evidence["summary"]


def orchestrator_json() -> str:
    return json.dumps(
        {
            "issue_id": "x/y#999",
            "suggested_labels": ["bug"],
            "triage_route": "bug",
            "next_steps": ["step"],
            "affected_modules": ["mod"],
            "similar_issues": [],
            "citations": ["CONTRIBUTING.md#Intro#0"],
        }
    )


def test_process_node_in_graph(tmp_path):
    tools = make_tools(tmp_path)
    process = ProcessAgent(model=ScriptedModel())
    historical = HistoricalAgent(model=HistoricalScriptedModel())

    async def run():
        async with AgentToolbox(tools) as box:
            graph = build_graph(
                toolbox=box,
                historical_agent=historical,
                process_agent=process,
                orchestrator_respond=lambda prompt: orchestrator_json(),
            ).compile()
            state = TriageState(
                issue="crash on empty frame",
                issue_id="x/y#999",
                classification={"type": "bug", "confidence": 0.9},
            )
            return await graph.ainvoke(state)

    result = asyncio.run(run())
    assert result["runbook_steps"]
    assert result["runbook_steps"][0]["path"] == "CONTRIBUTING.md"
    assert "CONTRIBUTING.md#Intro#0" in result["citations"]
