from triage.rag.embed import Embedder


def fake_embed_fn(texts):
    return [[float(len(t))] for t in texts]


def counting_embed_fn(calls):
    def embed(texts):
        calls["n"] += 1
        return [[float(len(t))] for t in texts]

    return embed


def test_embed_many_returns_in_order():
    embedder = Embedder(embed_fn=fake_embed_fn)
    assert embedder.embed_many(["a", "bb", "ccc"]) == [[1.0], [2.0], [3.0]]


def test_embed_single():
    embedder = Embedder(embed_fn=fake_embed_fn)
    assert embedder.embed("hello") == [5.0]


def test_embed_many_empty():
    embedder = Embedder(embed_fn=fake_embed_fn)
    assert embedder.embed_many([]) == []


def test_caches_within_session():
    calls = {"n": 0}
    embedder = Embedder(embed_fn=counting_embed_fn(calls))
    embedder.embed_many(["a", "b"])
    embedder.embed_many(["a", "b"])
    assert calls["n"] == 1


def test_only_uncached_texts_are_embedded():
    calls = {"n": 0}
    embedder = Embedder(embed_fn=counting_embed_fn(calls))
    embedder.embed_many(["a", "bb"])
    embedder.embed_many(["bb", "ccc"])
    assert calls["n"] == 2
    assert embedder.embed_many(["bb", "ccc"]) == [[2.0], [3.0]]


def test_cache_persists_to_disk(tmp_path):
    cache = tmp_path / "embeddings.json"
    calls = {"n": 0}
    embedder = Embedder(embed_fn=counting_embed_fn(calls), cache_path=cache)
    embedder.embed_many(["abc"])
    assert cache.exists()

    reloaded = Embedder(embed_fn=counting_embed_fn(calls), cache_path=cache)
    assert reloaded.embed_many(["abc"]) == [[3.0]]
    assert calls["n"] == 1
