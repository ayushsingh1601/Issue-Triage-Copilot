"""Graph nodes; specialist nodes are wired to the MCP toolbox via factories."""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from triage.agents.historical import HistoricalAgent
from triage.agents.process import ProcessAgent
from triage.mcp_tools.langchain import AgentToolbox
from triage.orchestration.state import TriageState


def plan_node(state: TriageState) -> dict:
    return {"classification": {"type": "", "confidence": 0.0}}


def historical_node(state: TriageState) -> dict:
    return {"historical_evidence": []}


def process_node(state: TriageState) -> dict:
    return {"runbook_steps": []}


def decide_node(state: TriageState) -> dict:
    return {"decision": None, "needs_human": False, "confidence": 0.0}


def human_in_loop_node(state: TriageState) -> dict:
    return {"needs_human": True}


def make_historical_node(
    toolbox: AgentToolbox,
    agent: HistoricalAgent | None = None,
) -> Callable[[TriageState], Awaitable[dict]]:
    agent = agent or HistoricalAgent()

    async def node(state: TriageState) -> dict:
        tools = await toolbox.group("historical")
        evidence = await agent.run(state.issue, tools)
        return {
            "historical_evidence": evidence["similar_issues"],
            "citations": evidence["citations"],
        }

    return node


def make_process_node(
    toolbox: AgentToolbox,
    agent: ProcessAgent | None = None,
) -> Callable[[TriageState], Awaitable[dict]]:
    agent = agent or ProcessAgent()

    async def node(state: TriageState) -> dict:
        tools = await toolbox.group("process")
        issue_type = state.classification.get("type", "")
        evidence = await agent.run(issue_type, state.issue, tools)
        return {
            "runbook_steps": evidence["steps"],
            "citations": evidence["citations"],
        }

    return node
