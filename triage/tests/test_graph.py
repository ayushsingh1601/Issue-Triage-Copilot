from triage.orchestration.edges import route_after_decide
from triage.orchestration.graph import build_graph
from triage.orchestration.state import TriageState


def test_graph_compiles():
    compiled = build_graph().compile()
    assert compiled is not None


def test_graph_runs_stub_end_to_end():
    state = TriageState(issue="crash on empty frame", issue_id="x/y#999")
    result = build_graph().compile().invoke(state)
    assert result["issue_id"] == "x/y#999"
    assert result["classification"] == {"type": "", "confidence": 0.0}
    assert result["decision"] is None
    assert result["needs_human"] is False


def test_route_after_decide():
    assert route_after_decide(TriageState(needs_human=False)) == "done"
    assert route_after_decide(TriageState(needs_human=True)) == "human"
