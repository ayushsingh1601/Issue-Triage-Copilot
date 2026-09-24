import numpy as np
from triage.rag.store import ChromaStore


def make_store(tmp_path, name: str = "issues") -> ChromaStore:
    return ChromaStore(path=tmp_path, collection=name)


def dim2(d):
    return [float(d), 0.0]


def test_add_and_count(tmp_path):
    store = make_store(tmp_path)
    store.add(
        ids=["a", "b"],
        texts=["apple", "banana"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"repo": "x"}, {"repo": "y"}],
    )
    assert store.count() == 2


def test_add_empty_is_noop(tmp_path):
    store = make_store(tmp_path)
    store.add(ids=[], texts=[], embeddings=[], metadatas=[])
    assert store.count() == 0


def test_query_returns_nearest_first(tmp_path):
    store = make_store(tmp_path)
    store.add(
        ids=["a", "b"],
        texts=["apple", "banana"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"repo": "x"}, {"repo": "y"}],
    )
    matches = store.query([0.9, 0.1], k=2)
    assert [m.id for m in matches] == ["a", "b"]
    assert matches[0].text == "apple"
    assert matches[0].metadata == {"repo": "x"}


def test_query_respects_k(tmp_path):
    store = make_store(tmp_path)
    store.add(
        ids=[str(i) for i in range(5)],
        texts=[f"text {i}" for i in range(5)],
        embeddings=[np.eye(2)[i % 2].tolist() for i in range(5)],
        metadatas=[{"repo": "x"}] * 5,
    )
    assert len(store.query([1.0, 0.0], k=3)) == 3


def test_query_metadata_filter(tmp_path):
    store = make_store(tmp_path)
    store.add(
        ids=["a", "b"],
        texts=["apple", "banana"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"repo": "x", "type": "bug"}, {"repo": "y", "type": "enhancement"}],
    )
    matches = store.query([1.0, 0.0], k=5, where={"repo": "x"})
    assert [m.id for m in matches] == ["a"]


def test_persistent_across_instances(tmp_path):
    store = make_store(tmp_path)
    store.add(ids=["a"], texts=["apple"], embeddings=[[1.0, 0.0]], metadatas=[{"repo": "x"}])
    reloaded = make_store(tmp_path)
    assert reloaded.count() == 1
    assert reloaded.query([1.0, 0.0], k=1)[0].id == "a"
