"""Shared ReAct loop for the specialist agents."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from triage.jsonutil import content_text


async def react_loop(
    model: Any,
    system: str,
    query: str,
    tools: list[BaseTool],
    max_steps: int = 5,
) -> list[Any]:
    by_name = {tool.name: tool for tool in tools}
    messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=query)]
    for _ in range(max_steps):
        response = await model.ainvoke(messages)
        if not response.tool_calls:
            messages.append(response)
            break
        for call in response.tool_calls:
            result = await by_name[call["name"]].ainvoke(call["args"])
            messages.append(
                ToolMessage(
                    content=content_text(result),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )
        messages.append(response)
    return messages
