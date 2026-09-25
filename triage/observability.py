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


def langfuse_callback() -> Any | None:
    if not langfuse_enabled():
        return None
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    client = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=os.environ.get("LANGFUSE_HOST", DEFAULT_HOST),
    )
    return CallbackHandler(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        trace_context=client,
    )


def graph_config() -> dict[str, Any]:
    """LangGraph invoke config carrying the Langfuse callback when configured."""
    callback = langfuse_callback()
    return {"callbacks": [callback]} if callback else {}
