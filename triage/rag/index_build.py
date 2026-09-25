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


def sync_issue_index(
    records: list[IssueRecord],
    embedder: Embedder,
    index_dir: Path,
    collection: str = "issues",
) -> dict[str, int]:
    """Incrementally sync the issue index: add new ids, delete evicted ids."""
    store = ChromaStore(index_dir / collection, collection)
    index_ids = set(store.ids())
    corpus_ids = {f"{r.repo}#{r.number}" for r in records}
    new_ids = corpus_ids - index_ids
    removed_ids = index_ids - corpus_ids
    if new_ids:
        new_records = [r for r in records if f"{r.repo}#{r.number}" in new_ids]
        chunks = [chunk_issue(r) for r in new_records]
        store.add(
            ids=[chunk.id for chunk in chunks],
            texts=[chunk.text for chunk in chunks],
            embeddings=embedder.embed_many([chunk.text for chunk in chunks]),
            metadatas=[
                {
                    "repo": chunk.repo,
                    "number": chunk.number,
                    "title": chunk.title,
                    "labels": ",".join(chunk.labels),
                }
                for chunk in chunks
            ],
        )
    if removed_ids:
        store.delete(sorted(removed_ids))
    return {
        "added": len(new_ids),
        "removed": len(removed_ids),
        "unchanged": len(index_ids & corpus_ids),
    }


def sync_doc_index(
    docs: list[ProcessDoc],
    embedder: Embedder,
    index_dir: Path,
    chunk_size: int = 600,
    overlap: float = 0.15,
    structural: bool = True,
    collection: str = "docs",
) -> dict[str, int]:
    """Rebuild the small doc collection in place."""
    store = ChromaStore(index_dir / collection, collection)
    store.clear()
    built = build_doc_index(
        docs,
        embedder,
        index_dir,
        chunk_size=chunk_size,
        overlap=overlap,
        structural=structural,
        collection=collection,
    )
    return {"chunks": built.count()}
