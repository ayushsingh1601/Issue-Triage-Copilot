"""Graph nodes; specialist nodes are wired to the MCP toolbox via factories."""
from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from typing import Any

from langchain_core.runnables import RunnableConfig
from triage.agents.historical import HistoricalAgent
from triage.agents.process import ProcessAgent
from triage.agents.react import invoke_respond
from triage.guardrails.citation_verify import verify_decision
from triage.guardrails.confidence import needs_human_review
from triage.guardrails.input_guard import GuardResult, InputGuard
from triage.guardrails.schema import TriageDecision
from triage.jsonutil import content_text, extract_json, parse_json_documents
from triage.mcp_tools.langchain import AgentToolbox
from triage.memory.store import EvidenceStore
from triage.orchestration.state import TriageState
from triage.prompts.orchestrator import build_orchestrator_prompt
from triage.tracing import Tracer

CONFIDENCE_THRESHOLD = 0.5


def default_orchestrator_respond() -> Any:
    from langchain_openai import ChatOpenAI

    model = os.environ.get("OPENAI_MAIN_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)

    class Respond:
        async def ainvoke(self, prompt: str, config: RunnableConfig = None) -> str:
            response = await llm.ainvoke(prompt, config=config)
            return response.content

    return Respond()


def plan_node(state: TriageState) -> dict:
    return {"classification": {"type": "", "confidence": 0.0}}


def make_guard_node(
    input_guard: InputGuard | None = None,
) -> Callable[[TriageState], dict]:
    def node(state: TriageState) -> dict:
        if input_guard is None:
            return {"guard_result": GuardResult(allowed=True, reason="no guard configured")}
        return {"guard_result": input_guard.guard(state.issue)}

    return node


def reject_node(state: TriageState) -> dict:
    return {"rejected": True, "needs_human": False, "decision": None, "confidence": 0.0}


def historical_node(state: TriageState) -> dict:
    return {"historical_evidence": []}


def process_node(state: TriageState) -> dict:
    return {"runbook_steps": []}


def decide_node(state: TriageState) -> dict:
    return {"decision": None, "needs_human": False, "confidence": 0.0}


def human_in_loop_node(state: TriageState) -> dict:
    return {"needs_human": True}


def make_plan_node(
    toolbox: AgentToolbox,
    tracer: Tracer | None = None,
) -> Callable[[TriageState, Any], Awaitable[dict]]:
    async def node(state: TriageState, config: RunnableConfig = None) -> dict:
        tools = await toolbox.group("orchestrator")
        classify = next(tool for tool in tools if tool.name == "classify_issue")
        with _maybe_span(tracer, "tool", tool="classify_issue"):
            result = await classify.ainvoke({"issue": state.issue}, config=config)
        classification = parse_json_documents(content_text(result))[0]
        return {"classification": classification}

    return node


def make_historical_node(
    toolbox: AgentToolbox,
    agent: HistoricalAgent | None = None,
    tracer: Tracer | None = None,
) -> Callable[[TriageState, Any], Awaitable[dict]]:
    agent = agent or HistoricalAgent()

    async def node(state: TriageState, config: RunnableConfig = None) -> dict:
        tools = await toolbox.group("historical")
        evidence = await agent.run(state.issue, tools, tracer=tracer, config=config)
        return {
            "historical_evidence": evidence["similar_issues"],
            "citations": evidence["citations"],
        }

    return node


def make_process_node(
    toolbox: AgentToolbox,
    agent: ProcessAgent | None = None,
    tracer: Tracer | None = None,
) -> Callable[[TriageState, Any], Awaitable[dict]]:
    agent = agent or ProcessAgent()

    async def node(state: TriageState, config: RunnableConfig = None) -> dict:
        tools = await toolbox.group("process")
        issue_type = state.classification.get("type", "")
        evidence = await agent.run(issue_type, state.issue, tools, tracer=tracer, config=config)
        return {
            "runbook_steps": evidence["steps"],
            "citations": evidence["citations"],
        }

    return node


def make_decide_node(
    toolbox: AgentToolbox,
    orchestrator_respond: Any | None = None,
    evidence_store: EvidenceStore | None = None,
    tracer: Tracer | None = None,
) -> Callable[[TriageState, Any], Awaitable[dict]]:
    orchestrator_respond = orchestrator_respond or default_orchestrator_respond()

    async def node(state: TriageState, config: RunnableConfig = None) -> dict:
        tools = await toolbox.group("orchestrator")
        patterns_tool = next(tool for tool in tools if tool.name == "get_resolution_patterns")
        issue_type = state.classification.get("type", "")
        with _maybe_span(tracer, "tool", tool="get_resolution_patterns"):
            result = await patterns_tool.ainvoke({"issue_type": issue_type}, config=config)
        patterns = parse_json_documents(content_text(result))[0]

        prompt = build_orchestrator_prompt(state, patterns)
        with _maybe_span(tracer, "llm"):
            raw = await invoke_respond(orchestrator_respond, prompt, config)
        decision = TriageDecision.model_validate_json(extract_json(raw))
        decision = verify_decision(
            decision,
            issue_ids=toolbox.issue_ids(),
            doc_ids=toolbox.doc_ids(),
        )

        confidence = float(state.classification.get("confidence", 0.0))
        needs_human = needs_human_review(confidence, CONFIDENCE_THRESHOLD)
        if evidence_store:
            evidence_store.put(
                state.issue_id,
                {
                    "classification": state.classification,
                    "historical_evidence": state.historical_evidence,
                    "runbook_steps": state.runbook_steps,
                    "decision": decision.model_dump(mode="json"),
                },
            )
        return {
            "decision": decision,
            "confidence": confidence,
            "needs_human": needs_human,
            "citations": decision.citations,
        }

    return node


def _maybe_span(tracer: Tracer | None, name: str, **detail):
    if tracer is None:
        return nullcontext()
    return tracer.span(name, **detail)
