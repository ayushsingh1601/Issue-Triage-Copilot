"""Index-building helpers shared by the corpus builder and the eval runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from triage.rag.chunk import chunk_doc, chunk_issue
from triage.rag.embed import Embedder
from triage.rag.parse import IssueRecord, ProcessDoc
from triage.rag.store import ChromaStore


def build_issue_index(
    records: list[IssueRecord],
    embedder: Embedder,
    index_dir: Path,
    collection: str = "issues",
) -> ChromaStore:
    store = ChromaStore(index_dir / collection, collection)
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for record in records:
        chunk = chunk_issue(record)
        ids.append(chunk.id)
        texts.append(chunk.text)
        metadatas.append(
            {
                "repo": chunk.repo,
                "number": chunk.number,
                "title": chunk.title,
                "labels": ",".join(chunk.labels),
            }
        )
    store.add(ids=ids, texts=texts, embeddings=embedder.embed_many(texts), metadatas=metadatas)
    return store


def build_doc_index(
    docs: list[ProcessDoc],
    embedder: Embedder,
    index_dir: Path,
    chunk_size: int = 600,
    overlap: float = 0.15,
    structural: bool = True,
    collection: str = "docs",
) -> ChromaStore:
    store = ChromaStore(index_dir / collection, collection)
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for doc in docs:
        for chunk in chunk_doc(doc, chunk_size=chunk_size, overlap=overlap, structural=structural):
            ids.append(chunk.id)
            texts.append(chunk.text)
            metadatas.append(
                {"repo": chunk.repo, "path": chunk.path, "heading_path": chunk.heading_path}
            )
    store.add(ids=ids, texts=texts, embeddings=embedder.embed_many(texts), metadatas=metadatas)
    return store
