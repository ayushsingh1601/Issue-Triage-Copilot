import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from triage.mcp_tools.langchain import TOOL_GROUPS, AgentToolbox
from triage.mcp_tools.tools import TriageTools
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
                body="fixed",
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


class FakeAgentModel:
    def __init__(self) -> None:
        self._calls = 0

    async def ainvoke(self, messages):
        self._calls += 1
        if self._calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "classify_issue",
                        "args": {"issue": "crash on empty frame"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="classified the issue")


async def run_react(model, tools, query: str) -> str:
    by_name = {tool.name: tool for tool in tools}
    messages = [HumanMessage(content=query)]
    for _ in range(3):
        response = await model.ainvoke(messages)
        if not response.tool_calls:
            return str(response.content)
        for call in response.tool_calls:
            result = await by_name[call["name"]].ainvoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        messages.append(response)
    return str(messages[-1].content)


def test_tool_groups_are_strict():
    assert TOOL_GROUPS == {
        "orchestrator": ["classify_issue", "get_resolution_patterns"],
        "historical": ["search_past_issues", "get_issue_details"],
        "process": ["get_runbook_steps"],
    }


def test_toolbox_exposes_per_agent_subsets(tmp_path):
    tools = make_tools(tmp_path)

    async def run():
        async with AgentToolbox(tools) as box:
            orchestrator = await box.group("orchestrator")
            historical = await box.group("historical")
            process = await box.group("process")
            return (
                sorted(t.name for t in orchestrator),
                sorted(t.name for t in historical),
                sorted(t.name for t in process),
            )

    orch, hist, proc = asyncio.run(run())
    assert orch == ["classify_issue", "get_resolution_patterns"]
    assert hist == ["get_issue_details", "search_past_issues"]
    assert proc == ["get_runbook_steps"]


def test_langchain_tool_call_end_to_end(tmp_path):
    tools = make_tools(tmp_path)

    async def run():
        async with AgentToolbox(tools) as box:
            historical = await box.group("historical")
            details_tool = next(t for t in historical if t.name == "get_issue_details")
            return await details_tool.ainvoke({"issue_id": "x/y#1"})

    result = asyncio.run(run())
    text = result[0]["text"]
    assert "Bug 1" in text
    assert "MEMBER" in text


def test_react_agent_calls_mcp_tool(tmp_path):
    tools = make_tools(tmp_path)

    async def run():
        async with AgentToolbox(tools) as box:
            orchestrator = await box.group("orchestrator")
            model = FakeAgentModel()
            final = await run_react(model, orchestrator, "classify this issue")
            return final, model._calls

    final, calls = asyncio.run(run())
    assert final == "classified the issue"
    assert calls == 2
