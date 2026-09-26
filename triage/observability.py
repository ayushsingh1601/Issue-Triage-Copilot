"""Optional Langfuse tracing (open-source; self-hosted or free cloud tier).

Langfuse is only engaged when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set;
otherwise these helpers return no-ops so the project still runs with just the local
`triage.tracing.Tracer`.
"""
from __future__ import annotations

import os
from typing import Any

DEFAULT_HOST = "https://cloud.langfuse.com"


def langfuse_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def _langfuse_host() -> str:
    return os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or DEFAULT_HOST


def langfuse_callback() -> Any | None:
    if not langfuse_enabled():
        return None
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=_langfuse_host(),
    )
    return CallbackHandler(public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"))


def graph_config() -> dict[str, Any]:
    """LangGraph invoke config carrying the Langfuse callback when configured."""
    callback = langfuse_callback()
    return {"callbacks": [callback]} if callback else {}
