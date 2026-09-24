from triage.rag.chunk import chunk_doc, chunk_issue, count_tokens
from triage.rag.parse import IssueRecord, ProcessDoc

DOC = (
    "# Intro\n"
    "paragraph one intro text.\n"
    "## Setup\n"
    "setup steps here.\n"
    "### Install\n"
    "pip install thing.\n"
    "more install text.\n"
    "# Other\n"
    "other content here.\n"
)


def make_issue(body: str = "body", title: str = "Bug: crash", number: int = 42) -> IssueRecord:
    return IssueRecord(
        repo="x/y",
        number=number,
        title=title,
        body=body,
        state="closed",
        author="a",
        created_at="",
        closed_at=None,
        labels=["bug"],
        linked_prs=[],
        comments=[],
    )


def test_count_tokens():
    assert count_tokens("hello") == 1


def test_chunk_issue_keeps_title_body_and_code_fences():
    body = "Repro:\n```\nTraceback (most recent call last):\n    raise ValueError\n```\n"
    chunk = chunk_issue(make_issue(body=body), max_tokens=10000)
    assert chunk.id == "x/y#42"
    assert chunk.repo == "x/y"
    assert chunk.number == 42
    assert chunk.labels == ["bug"]
    assert chunk.title in chunk.text
    assert "Traceback" in chunk.text
    assert "```" in chunk.text


def test_chunk_issue_truncates_to_budget():
    body = " ".join(["word"] * 5000)
    chunk = chunk_issue(make_issue(body=body), max_tokens=100)
    assert count_tokens(chunk.text) <= 100
    assert chunk.title in chunk.text


def test_chunk_doc_structural_headings():
    doc = ProcessDoc(repo="x/y", path="CONTRIBUTING.md", content=DOC)
    chunks = chunk_doc(doc, chunk_size=1000, overlap=0.1)
    paths = {c.heading_path for c in chunks}
    assert {"Intro", "Intro > Setup", "Intro > Setup > Install", "Other"} <= paths
    for para in [
        "paragraph one intro text.",
        "setup steps here.",
        "pip install thing.",
        "other content here.",
    ]:
        assert any(para in c.text for c in chunks)


def test_chunk_doc_flat_has_no_heading_paths():
    doc = ProcessDoc(repo="x/y", path="docs/a.md", content=DOC)
    chunks = chunk_doc(doc, chunk_size=100, overlap=0.1, structural=False)
    assert chunks
    assert all(c.heading_path == "" for c in chunks)
    for para in ["paragraph one intro text.", "setup steps here.", "other content here."]:
        assert any(para in c.text for c in chunks)


def test_chunk_doc_respects_chunk_size():
    content = "word " * 3000
    for structural in (True, False):
        doc = ProcessDoc(repo="x/y", path="docs/a.md", content=content)
        chunks = chunk_doc(doc, chunk_size=100, overlap=0.15, structural=structural)
        assert chunks
        assert all(count_tokens(c.text) <= 100 for c in chunks)


def test_chunk_doc_overlap_increases_total_tokens():
    doc = ProcessDoc(repo="x/y", path="docs/a.md", content="word " * 3000)
    no_overlap = chunk_doc(doc, chunk_size=100, overlap=0.0)
    with_overlap = chunk_doc(doc, chunk_size=100, overlap=0.15)
    total_no = sum(count_tokens(c.text) for c in no_overlap)
    total_yes = sum(count_tokens(c.text) for c in with_overlap)
    assert total_yes > total_no
    assert len(with_overlap) >= len(no_overlap)


def test_chunk_doc_empty():
    doc = ProcessDoc(repo="x/y", path="docs/a.md", content="")
    assert chunk_doc(doc) == []
