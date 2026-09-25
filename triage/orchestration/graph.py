"""LangGraph skeleton: orchestrator plans, fans out to specialists in parallel,
re-enters to decide, and routes to a human-in-the-loop edge on low confidence."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from triage.orchestration.edges import route_after_decide
from triage.orchestration.nodes import (
    decide_node,
    historical_node,
    human_in_loop_node,
    plan_node,
    process_node,
)
from triage.orchestration.state import TriageState


def build_graph() -> StateGraph:
    graph = StateGraph(TriageState)
    graph.add_node("plan", plan_node)
    graph.add_node("historical", historical_node)
    graph.add_node("process", process_node)
    graph.add_node("decide", decide_node)
    graph.add_node("human_in_loop", human_in_loop_node)
    graph.add_edge(START, "plan")
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
