"""Conditional edges for the triage graph."""
from __future__ import annotations

from triage.orchestration.state import TriageState


def route_after_decide(state: TriageState) -> str:
    if state.needs_human:
        return "human"
    return "done"
