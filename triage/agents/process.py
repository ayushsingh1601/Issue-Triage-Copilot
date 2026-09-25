"""Process agent: runbook steps from process docs via ReAct over MCP."""
from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from triage.agents.react import react_loop
from triage.jsonutil import parse_json_documents
from triage.prompts.process import SYSTEM_PROMPT
from triage.tracing import Tracer


def default_model() -> Any:
    from langchain_openai import ChatOpenAI

    model = os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)


class ProcessAgent:
    def __init__(self, model: Any | None = None) -> None:
        self._model = model or default_model()

    async def run(
        self,
        issue_type: str,
        query: str,
        tools: list[BaseTool],
        tracer: Tracer | None = None,
    ) -> dict[str, Any]:
        prompt = f"Issue type: {issue_type}\n\nIssue:\n{query}"
        transcript = await react_loop(self._model, SYSTEM_PROMPT, prompt, tools, tracer=tracer)
        evidence: dict[str, Any] = {"steps": [], "citations": [], "summary": ""}
        citations: list[str] = []
        for message in transcript:
            if isinstance(message, ToolMessage) and message.name == "get_runbook_steps":
                for item in parse_json_documents(message.content):
                    evidence["steps"].append(item)
                    if item["source"] not in citations:
                        citations.append(item["source"])
            elif isinstance(message, AIMessage) and not message.tool_calls:
                evidence["summary"] = message.content
        evidence["citations"] = citations
        return evidence
