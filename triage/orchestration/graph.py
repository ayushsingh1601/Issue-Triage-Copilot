"""LangGraph graph: orchestrator plans, fans out to specialists in parallel,
re-enters to decide, and routes to a human-in-the-loop edge on low confidence."""
from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph
from triage.agents.historical import HistoricalAgent
from triage.agents.process import ProcessAgent
from triage.mcp_tools.langchain import AgentToolbox
from triage.memory.store import EvidenceStore
from triage.orchestration.edges import route_after_decide
from triage.orchestration.nodes import (
    decide_node,
    historical_node,
    human_in_loop_node,
    make_decide_node,
    make_historical_node,
    make_plan_node,
    make_process_node,
    plan_node,
    process_node,
)
from triage.orchestration.state import TriageState


def build_graph(
    toolbox: AgentToolbox | None = None,
    historical_agent: HistoricalAgent | None = None,
    process_agent: ProcessAgent | None = None,
    orchestrator_respond: Callable[[str], str] | None = None,
    evidence_store: EvidenceStore | None = None,
) -> StateGraph:
    graph = StateGraph(TriageState)
    if toolbox:
        graph.add_node("plan", make_plan_node(toolbox))
        graph.add_node("historical", make_historical_node(toolbox, historical_agent))
        graph.add_node("process", make_process_node(toolbox, process_agent))
        graph.add_node("decide", make_decide_node(toolbox, orchestrator_respond, evidence_store))
    else:
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
