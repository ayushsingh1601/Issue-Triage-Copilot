import asyncio

from triage.mcp_tools.server import create_server
from triage.mcp_tools.tools import TriageTools
from triage.persist import save_records
from triage.rag.parse import IssueComment, IssueRecord
from triage.rag.store import ChromaStore


def make_issue(
    repo: str,
    number: int,
    labels: list[str],
    body: str = "crash on empty frame with stack trace",
    comments: list[IssueComment] | None = None,
    linked_prs: list[int] | None = None,
) -> IssueRecord:
    return IssueRecord(
        repo=repo,
        number=number,
        title=f"Bug {number}",
        body=body,
        state="closed",
        author="a",
        created_at="2024-01-01",
        closed_at="2024-01-02",
        labels=labels,
        linked_prs=linked_prs or [],
        comments=comments or [],
    )


def make_tools(tmp_path) -> TriageTools:
    def fake_embed(texts):
        return [[float(i + 1), 0.0] for i, _ in enumerate(texts)]

    def fake_llm(prompt):
        return '{"type": "bug", "confidence": 0.9}'

    records = [
        make_issue(
            "x/y",
            1,
            labels=["bug", "DataFrame"],
            comments=[
                IssueComment(
                    id=1,
                    author="maint",
                    author_association="MEMBER",
                    body="fixed in #45",
                    created_at="2024-01-02",
                )
            ],
            linked_prs=[45],
        ),
        make_issue("x/y", 2, labels=["bug"]),
        make_issue("x/y", 3, labels=["enhancement"]),
        make_issue("a/b", 1, labels=["bug"]),
    ]
    save_records(records, tmp_path / "processed" / "issues_corpus.json")

    issue_store = ChromaStore(tmp_path / "indexes" / "issues", "issues")
    issue_store.add(
        ids=[f"{r.repo}#{r.number}" for r in records],
        texts=[f"{r.title}\n\n{r.body}" for r in records],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        metadatas=[{"repo": r.repo, "number": r.number, "title": r.title} for r in records],
    )
    doc_store = ChromaStore(tmp_path / "indexes" / "docs", "docs")
    doc_store.add(
        ids=["CONTRIBUTING.md#top#0"],
        texts=["# Contributing\nReport bugs with a minimal repro and full stack trace."],
        embeddings=[[1.5, 0.0]],
        metadatas=[{"repo": "x/y", "path": "CONTRIBUTING.md", "heading_path": ""}],
    )
    return TriageTools(
        indexes_dir=tmp_path / "indexes",
        processed_dir=tmp_path / "processed",
        embed_fn=fake_embed,
        llm_respond=fake_llm,
    )


def test_classify_issue(tmp_path):
    tools = make_tools(tmp_path)
    result = tools.classify_issue("crash when reading empty csv")
    assert result == {"type": "bug", "confidence": 0.9}


def test_search_past_issues(tmp_path):
    tools = make_tools(tmp_path)
    results = tools.search_past_issues("crash on empty frame", limit=3)
    assert results
    assert results[0]["issue_id"] == "x/y#1"
    assert "bug" in results[0]["labels"]
    assert results[0]["linked_prs"] == [45]
    assert results[0]["closing_comments"] == ["fixed in #45"]


def test_search_past_issues_filters_by_repo_and_type(tmp_path):
    tools = make_tools(tmp_path)
    by_repo = tools.search_past_issues("crash", repo="a/b", limit=5)
    assert all(r["issue_id"].startswith("a/b#") for r in by_repo)
    by_type = tools.search_past_issues("crash", issue_type="enhancement", limit=5)
    assert by_type and all("enhancement" in r["labels"] for r in by_type)


def test_get_issue_details(tmp_path):
    tools = make_tools(tmp_path)
    details = tools.get_issue_details("x/y#1")
    assert details is not None
    assert details["title"] == "Bug 1"
    assert details["comments"][0]["author_association"] == "MEMBER"
    assert details["linked_prs"] == [45]


def test_get_issue_details_unknown(tmp_path):
    tools = make_tools(tmp_path)
    assert tools.get_issue_details("x/y#999") is None


def test_get_runbook_steps(tmp_path):
    tools = make_tools(tmp_path)
    steps = tools.get_runbook_steps("bug", "how to report")
    assert steps
    assert steps[0]["path"] == "CONTRIBUTING.md"
    assert "stack trace" in steps[0]["step"]


def test_get_resolution_patterns(tmp_path):
    tools = make_tools(tmp_path)
    patterns = tools.get_resolution_patterns("bug")
    assert patterns["count"] == 3
    assert "DataFrame" in patterns["co_occurring_labels"]
    assert patterns["pr_link_rate"] == round(1 / 3, 3)
    assert patterns["first_response_associations"] == {"MEMBER": 1}


def test_get_resolution_patterns_empty(tmp_path):
    tools = make_tools(tmp_path)
    patterns = tools.get_resolution_patterns("documentation")
    assert patterns["count"] == 0
    assert patterns["co_occurring_labels"] == []


def test_mcp_server_calls_tools(tmp_path):
    tools = make_tools(tmp_path)
    server = create_server(tools)
    run = asyncio.run

    def text(content):
        return content[0][0].text

    labels = run(server.call_tool("classify_issue", {"issue": "crash"}))
    assert "bug" in text(labels)
    details = run(server.call_tool("get_issue_details", {"issue_id": "x/y#1"}))
    assert "Bug 1" in text(details)
    patterns = run(server.call_tool("get_resolution_patterns", {"issue_type": "bug"}))
    assert '"count": 3' in text(patterns)


def test_stale_index_warns(tmp_path, capsys):
    from triage.mcp_tools.tools import TriageTools

    def fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    record = make_issue("x/y", 1, labels=["bug"])
    save_records([record], tmp_path / "processed" / "issues_corpus.json")
    store = ChromaStore(tmp_path / "indexes" / "issues", "issues")
    store.add(
        ids=["a/b#99"],
        texts=["unrelated"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"repo": "a/b"}],
    )
    doc_store = ChromaStore(tmp_path / "indexes" / "docs", "docs")
    doc_store.add(
        ids=["CONTRIBUTING.md#top#0"],
        texts=["# Contributing"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"repo": "x/y", "path": "CONTRIBUTING.md", "heading_path": ""}],
    )
    tools = TriageTools(
        indexes_dir=tmp_path / "indexes",
        processed_dir=tmp_path / "processed",
        embed_fn=fake_embed,
        llm_respond=lambda prompt: '{"type": "bug", "confidence": 0.9}',
    )
    captured = capsys.readouterr()
    assert "index is stale" in captured.err
    assert tools.index_stats()["missing"] == 1
