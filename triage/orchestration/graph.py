"""LangGraph graph: orchestrator plans, fans out to specialists in parallel,
re-enters to decide, and routes to a human-in-the-loop edge on low confidence."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from triage.agents.historical import HistoricalAgent
from triage.agents.process import ProcessAgent
from triage.guardrails.input_guard import InputGuard
from triage.mcp_tools.langchain import AgentToolbox
from triage.memory.store import EvidenceStore
from triage.orchestration.edges import route_after_decide, route_after_guard
from triage.orchestration.nodes import (
    decide_node,
    historical_node,
    human_in_loop_node,
    make_decide_node,
    make_guard_node,
    make_historical_node,
    make_plan_node,
    make_process_node,
    plan_node,
    process_node,
    reject_node,
)
from triage.orchestration.state import TriageState
from triage.tracing import Tracer


def build_graph(
    toolbox: AgentToolbox | None = None,
    historical_agent: HistoricalAgent | None = None,
    process_agent: ProcessAgent | None = None,
    orchestrator_respond: Callable[[str], str] | None = None,
    evidence_store: EvidenceStore | None = None,
    tracer: Tracer | None = None,
    input_guard: InputGuard | None = None,
) -> StateGraph:
    graph = StateGraph(TriageState)
    graph.add_node("guard", make_guard_node(input_guard))
    graph.add_node("reject", reject_node)
    if toolbox:
        graph.add_node("plan", _timed("node:plan", tracer, make_plan_node(toolbox, tracer)))
        graph.add_node(
            "historical",
            _timed(
                "node:historical",
                tracer,
                make_historical_node(toolbox, historical_agent, tracer),
            ),
        )
        graph.add_node(
            "process",
            _timed(
                "node:process",
                tracer,
                make_process_node(toolbox, process_agent, tracer),
            ),
        )
        graph.add_node(
            "decide",
            _timed(
                "node:decide",
                tracer,
                make_decide_node(toolbox, orchestrator_respond, evidence_store, tracer),
            ),
        )
    else:
        graph.add_node("plan", plan_node)
        graph.add_node("historical", historical_node)
        graph.add_node("process", process_node)
        graph.add_node("decide", decide_node)
    graph.add_node("human_in_loop", human_in_loop_node)
    graph.add_edge(START, "guard")
    graph.add_conditional_edges(
        "guard",
        route_after_guard,
        {"plan": "plan", "reject": "reject"},
    )
    graph.add_edge("reject", END)
    graph.add_edge("plan", "historical")
    graph.add_edge("plan", "process")
    graph.add_edge("historical", "decide")
    graph.add_edge("process", "decide")
    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {"human": "human_in_loop", "done": END},
    )
    graph.add_edge("human_in_loop", END)
    return graph


def _timed(name: str, tracer: Tracer | None, fn) -> Callable[[TriageState, Any], Awaitable[dict]]:
    async def wrapped(state: TriageState, config: RunnableConfig = None) -> dict:
        if tracer is None:
            return await fn(state, config)
        with tracer.span(name):
            return await fn(state, config)

    return wrapped
