"""Shared ReAct loop for the specialist agents."""
from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from triage.jsonutil import content_text
from triage.tracing import Tracer


async def react_loop(
    model: Any,
    system: str,
    query: str,
    tools: list[BaseTool],
    max_steps: int = 5,
    tracer: Tracer | None = None,
) -> list[Any]:
    by_name = {tool.name: tool for tool in tools}
    messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=query)]
    for _ in range(max_steps):
        start = time.perf_counter()
        response = await model.ainvoke(messages)
        if tracer:
            tracer.record("llm", (time.perf_counter() - start) * 1000)
        if not response.tool_calls:
            messages.append(response)
            break
        for call in response.tool_calls:
            start = time.perf_counter()
            result = await by_name[call["name"]].ainvoke(call["args"])
            if tracer:
                tracer.record("tool", (time.perf_counter() - start) * 1000, tool=call["name"])
            messages.append(
                ToolMessage(
                    content=content_text(result),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )
        messages.append(response)
    return messages
