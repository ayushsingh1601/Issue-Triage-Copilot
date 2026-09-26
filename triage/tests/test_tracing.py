import asyncio
import json

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from triage.agents.historical import HistoricalAgent
from triage.agents.process import ProcessAgent
from triage.mcp_tools.langchain import AgentToolbox
from triage.mcp_tools.tools import TriageTools
from triage.orchestration.graph import build_graph
from triage.orchestration.state import TriageState
from triage.persist import save_records
from triage.rag.parse import IssueComment, IssueRecord
from triage.rag.store import ChromaStore
from triage.tracing import Tracer


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class ToolCallModel:
    def __init__(self) -> None:
        self._steps = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "add",
                        "args": {"a": 1, "b": 2},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="3"),
        ]
        self._index = 0

    async def ainvoke(self, messages, config=None):
        step = self._steps[self._index % len(self._steps)]
        self._index += 1
        return step


def test_tracer_spans_and_records():
    tracer = Tracer()
    with tracer.span("sleep"):
        pass
    tracer.record("tool", 1.0, tool="classify_issue")
    events = tracer.events()
    assert {e["name"] for e in events} == {"sleep", "tool"}
    assert events[1]["tool"] == "classify_issue"


def test_tracer_summary():
    tracer = Tracer()
    tracer.record("llm", 10)
    tracer.record("llm", 20)
    tracer.record("tool", 5)
    summary = tracer.summary()
    assert summary["llm"] == {"count": 2, "avg_ms": 15.0, "p95_ms": 20.0}
    assert summary["tool"]["count"] == 1


def test_tracer_save_round_trip(tmp_path):
    path = tmp_path / "trace.json"
    tracer = Tracer()
    tracer.record("llm", 3)
    tracer.save(path)
    assert json.loads(path.read_text()) == tracer.events()


async def run_loop():
    from triage.agents.react import react_loop

    tracer = Tracer()
    await react_loop(
        ToolCallModel(), "system", "add 1 and 2", [add], tracer=tracer
    )
    return tracer


def test_react_loop_records_llm_and_tool_latency():
    tracer = asyncio.run(run_loop())
    names = {event["name"] for event in tracer.events()}
    assert "llm" in names
    assert "tool" in names
    tool_event = next(e for e in tracer.events() if e["name"] == "tool")
    assert tool_event["tool"] == "add"


def make_tools(tmp_path) -> TriageTools:
    def fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    def fake_llm(prompt):
        return json.dumps({"type": "bug", "confidence": 0.9})

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
        comments=[
            IssueComment(
                id=1, author="m", author_association="MEMBER", body="fixed", created_at="2024-01-02"
            )
        ],
    )
    save_records([record], tmp_path / "processed" / "issues_corpus.json")
    issue_store = ChromaStore(tmp_path / "indexes" / "issues", "issues")
    issue_store.add(
        ids=["x/y#1"],
        texts=["Bug 1\n\ncrash"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"repo": "x/y", "number": 1}],
    )
    doc_store = ChromaStore(tmp_path / "indexes" / "docs", "docs")
    doc_store.add(
        ids=["CONTRIBUTING.md#top#0"],
        texts=["# Contributing\nreport bugs."],
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
            AIMessage(content="similar crash"),
        ]
        self._index = 0

    async def ainvoke(self, messages, config=None):
        step = self._steps[self._index % len(self._steps)]
        self._index += 1
        return step


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
            AIMessage(content="runbook"),
        ]
        self._index = 0

    async def ainvoke(self, messages, config=None):
        step = self._steps[self._index % len(self._steps)]
        self._index += 1
        return step


def orchestrator_json() -> str:
    return json.dumps(
        {
            "issue_id": "x/y#99",
            "suggested_labels": ["bug"],
            "triage_route": "bug",
            "next_steps": ["step"],
            "affected_modules": ["mod"],
            "similar_issues": [],
            "citations": ["x/y#1"],
        }
    )


def test_graph_records_node_latencies(tmp_path):
    tools = make_tools(tmp_path)
    tracer = Tracer()
    historical = HistoricalAgent(model=HistoricalScriptedModel())
    process = ProcessAgent(model=ProcessScriptedModel())

    async def run():
        async with AgentToolbox(tools) as box:
            graph = build_graph(
                toolbox=box,
                historical_agent=historical,
                process_agent=process,
                orchestrator_respond=lambda prompt: orchestrator_json(),
                tracer=tracer,
            ).compile()
            await graph.ainvoke(
                TriageState(
                    issue="crash",
                    issue_id="x/y#99",
                    classification={"type": "bug", "confidence": 0.9},
                )
            )

    asyncio.run(run())
    names = {event["name"] for event in tracer.events()}
    assert {"node:plan", "node:historical", "node:process", "node:decide"} <= names
    assert "tool" in names
    assert "llm" in names


class RecordingCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.llm_ends: list = []
        self.tool_ends: list = []
        self.chain_ends: list = []

    def on_llm_end(self, response, **kwargs) -> None:
        self.llm_ends.append(response)

    def on_tool_end(self, output, **kwargs) -> None:
        self.tool_ends.append(output)

    def on_chain_end(self, output, **kwargs) -> None:
        self.chain_ends.append(output)


def test_config_threads_callbacks_to_agent_calls(tmp_path):
    tools = make_tools(tmp_path)
    historical = HistoricalAgent(
        model=FakeMessagesListChatModel(
            responses=[
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
                AIMessage(content="similar crash in x/y#1"),
            ]
        )
    )
    process = ProcessAgent(
        model=FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_runbook_steps",
                            "args": {"issue_type": "bug", "query": "crash"},
                            "id": "call_2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="runbook step from CONTRIBUTING.md"),
            ]
        )
    )
    recorder = RecordingCallbackHandler()

    async def run():
        async with AgentToolbox(tools) as box:
            graph = build_graph(
                toolbox=box,
                historical_agent=historical,
                process_agent=process,
                orchestrator_respond=lambda prompt: orchestrator_json(),
            ).compile()
            return await graph.ainvoke(
                TriageState(
                    issue="crash",
                    issue_id="x/y#99",
                    classification={"type": "bug", "confidence": 0.9},
                ),
                config={"callbacks": [recorder]},
            )

    result = asyncio.run(run())
    assert recorder.llm_ends, "specialist LLM outputs were not captured"
    assert recorder.tool_ends, "tool outputs were not captured"
    assert any(isinstance(out, dict) and "decision" in out for out in recorder.chain_ends), (
        "final decision not captured in the trace"
    )
    assert result["decision"] is not None
