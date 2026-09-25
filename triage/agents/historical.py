"""Historical-context ReAct agent: similar resolved issues and resolution patterns."""
from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from triage.agents.react import react_loop
from triage.jsonutil import parse_json_documents

SYSTEM_PROMPT = (
    "You are a historical-context specialist for issue triage. "
    "Use search_past_issues to find similar resolved issues, then get_issue_details "
    "to inspect the most relevant threads. Ground your summary in cited issue IDs "
    "like repo#number."
)


def default_model() -> Any:
    from langchain_openai import ChatOpenAI

    model = os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)


class HistoricalAgent:
    def __init__(self, model: Any | None = None) -> None:
        self._model = model or default_model()

    async def run(self, issue: str, tools: list[BaseTool]) -> dict[str, Any]:
        transcript = await react_loop(self._model, SYSTEM_PROMPT, issue, tools)
        evidence: dict[str, Any] = {
            "similar_issues": [],
            "details": [],
            "citations": [],
            "summary": "",
        }
        citations: list[str] = []
        for message in transcript:
            if isinstance(message, ToolMessage) and message.name == "search_past_issues":
                for item in parse_json_documents(message.content):
                    evidence["similar_issues"].append(item)
                    _add_citation(citations, item["issue_id"])
            elif isinstance(message, ToolMessage) and message.name == "get_issue_details":
                data = parse_json_documents(message.content)[0]
                evidence["details"].append(data)
                if data and data.get("issue_id"):
                    _add_citation(citations, data["issue_id"])
            elif isinstance(message, AIMessage) and not message.tool_calls:
                evidence["summary"] = message.content
        evidence["citations"] = citations
        return evidence


def _add_citation(citations: list[str], issue_id: str) -> None:
    if issue_id not in citations:
        citations.append(issue_id)
