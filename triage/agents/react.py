"""Shared ReAct loop for the specialist agents."""
from __future__ import annotations

import inspect
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
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
    config: RunnableConfig = None,
) -> list[Any]:
    """ReAct loop with function calling.

    The model is bound to the tools before the loop; without this the LLM
    never sees the tool schemas and returns a plain answer (no evidence).
    Scripted/fake models used in tests are used as-is when binding is not
    supported.
    """
    if hasattr(model, "bind_tools"):
        try:
            model = model.bind_tools(tools)
        except NotImplementedError:
            pass
    by_name = {tool.name: tool for tool in tools}
    messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=query)]
    for _ in range(max_steps):
        start = time.perf_counter()
        response = await model.ainvoke(messages, config=config)
        if tracer:
            tracer.record("llm", (time.perf_counter() - start) * 1000)
        messages.append(response)
        if not response.tool_calls:
            break
        for call in response.tool_calls:
            start = time.perf_counter()
            result = await by_name[call["name"]].ainvoke(call["args"], config=config)
            if tracer:
                tracer.record("tool", (time.perf_counter() - start) * 1000, tool=call["name"])
            messages.append(
                ToolMessage(
                    content=content_text(result),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )
    return messages


async def invoke_respond(
    respond: Any,
    prompt: str,
    config: RunnableConfig = None,
) -> str:
    """Invoke an LLM respond callable, threading `config` when it supports it."""
    if hasattr(respond, "ainvoke"):
        return await respond.ainvoke(prompt, config=config)
    return respond(prompt)


def invoke_respond_sync(
    respond: Any,
    prompt: str,
    config: RunnableConfig = None,
) -> str:
    """Sync variant that threads `config` only into callables that accept it."""
    if config is None:
        return respond(prompt)
    try:
        signature = inspect.signature(respond)
    except (TypeError, ValueError):
        return respond(prompt)
    if "config" in signature.parameters:
        return respond(prompt, config=config)
    return respond(prompt)
