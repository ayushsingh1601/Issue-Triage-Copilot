import pytest
from triage.observability import graph_config, langfuse_callback, langfuse_enabled


def test_langfuse_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    assert langfuse_enabled() is False
    assert langfuse_callback() is None
    assert graph_config() == {}


def test_langfuse_enabled_creates_callback(monkeypatch):
    pytest.importorskip("langfuse")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    assert langfuse_enabled() is True
    callback = langfuse_callback()
    assert callback is not None
    config = graph_config()
    assert config["callbacks"] and config["callbacks"][0] is not None
