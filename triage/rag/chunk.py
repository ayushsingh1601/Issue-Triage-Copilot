"""Chunking: whole-issue units for index A, heading-aware + recursive for index B."""
from __future__ import annotations

import re

import tiktoken
from pydantic import BaseModel
from triage.rag.parse import IssueRecord, ProcessDoc

_ENC = tiktoken.get_encoding("cl100k_base")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SEPARATORS = ["\n\n", "\n", " "]


class IssueChunk(BaseModel):
    id: str
    repo: str
    number: int
    title: str
    labels: list[str]
    text: str


class DocChunk(BaseModel):
    id: str
    repo: str
    path: str
    heading_path: str
    text: str


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def chunk_issue(record: IssueRecord, max_tokens: int = 800) -> IssueChunk:
    return IssueChunk(
        id=f"{record.repo}#{record.number}",
        repo=record.repo,
        number=record.number,
        title=record.title,
        labels=record.labels,
        text=issue_signature(record, max_tokens),
    )


def issue_signature(record: IssueRecord, max_tokens: int = 800) -> str:
    text = f"{record.title}\n\n{record.body}"
    tokens = _ENC.encode(text)
    if len(tokens) > max_tokens:
        text = _ENC.decode(tokens[:max_tokens])
    return text


def chunk_doc(
    doc: ProcessDoc,
    chunk_size: int = 600,
    overlap: float = 0.15,
    structural: bool = True,
) -> list[DocChunk]:
    overlap_tokens = int(chunk_size * overlap)
    sections = _split_sections(doc.content) if structural else [("", doc.content)]
    chunks: list[DocChunk] = []
    for heading_path, section in sections:
        pieces = _atomic_split(section, chunk_size)
        for index, text in enumerate(_merge(pieces, chunk_size, overlap_tokens)):
            chunks.append(
                DocChunk(
                    id=f"{doc.repo}:{doc.path}#{heading_path or 'top'}#{index}",
                    repo=doc.repo,
                    path=doc.path,
                    heading_path=heading_path,
                    text=text,
                )
            )
    return chunks


def _split_sections(content: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    heading = ""
    lines: list[str] = []
    for line in content.splitlines():
        match = _HEADING.match(line)
        if match:
            if lines:
                sections.append((heading, "\n".join(lines)))
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2)))
            heading = " > ".join(title for _, title in stack)
            lines = [line]
        else:
            lines.append(line)
    if lines:
        sections.append((heading, "\n".join(lines)))
    return sections


def _atomic_split(text: str, chunk_size: int) -> list[str]:
    if count_tokens(text) <= chunk_size:
        return [text]
    for sep in _SEPARATORS:
        parts = text.split(sep)
        if len(parts) > 1:
            pieces: list[str] = []
            for i, part in enumerate(parts):
                suffix = sep if i < len(parts) - 1 else ""
                pieces.extend(_atomic_split(part + suffix, chunk_size))
            return pieces
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _merge(pieces: list[str], chunk_size: int, overlap_tokens: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif count_tokens(current + piece) <= chunk_size:
            current += piece
        else:
            chunks.append(current)
            candidate = _tail_tokens(current, overlap_tokens) + piece
            current = candidate if count_tokens(candidate) <= chunk_size else piece
    if current:
        chunks.append(current)
    return chunks


def _tail_tokens(text: str, n: int) -> str:
    if n <= 0:
        return ""
    return _ENC.decode(_ENC.encode(text)[-n:])
