"""Embedding layer over OpenAI text-embedding-3-small with on-disk caching."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

DEFAULT_MODEL = "text-embedding-3-small"

EmbedFn = Callable[[list[str]], list[list[float]]]


def default_embed_fn(model: str = DEFAULT_MODEL) -> EmbedFn:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model).embed_documents


class Embedder:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_path: Path | str | None = None,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        self._model = model
        self._cache_path = Path(cache_path) if cache_path else None
        self._embed_fn = embed_fn or default_embed_fn(model)
        self._cache: dict[str, list[float]] = {}
        if self._cache_path and self._cache_path.exists():
            self._cache = json.loads(self._cache_path.read_text())

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for i, text in enumerate(texts):
            key = _hash(text)
            if key in self._cache:
                vectors[i] = self._cache[key]
            else:
                missing.append(i)
        if missing:
            results = self._embed_fn([texts[i] for i in missing])
            for i, vector in zip(missing, results, strict=True):
                vectors[i] = vector
                self._cache[_hash(texts[i])] = vector
            self._persist()
        return [v for v in vectors if v is not None]

    def _persist(self) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache))


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
