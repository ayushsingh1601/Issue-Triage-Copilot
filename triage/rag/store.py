"""ChromaDB wrapper for similarity search with metadata filters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from pydantic import BaseModel


class Match(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class ChromaStore:
    def __init__(self, path: Path | str, collection: str) -> None:
        self._collection = chromadb.PersistentClient(path=str(path)).get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self._collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[Match]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            Match(id=id_, text=document, metadata=metadata, distance=distance)
            for id_, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    def count(self) -> int:
        return self._collection.count()

    def ids(self) -> list[str]:
        return self._collection.get(include=[])["ids"]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._collection.delete(ids=ids)

    def clear(self) -> None:
        ids = self._collection.get(include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
