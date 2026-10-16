"""Graph wiring for Part 1 (planning).

    START → clarify_detect ⇄ clarify_ask
          → [mode router]
              requirements_first: scope_draft → feasibility → scope_review
              design_first:       design (subgraph: design_draft → feasibility → design_review)
          → draft_srd → testability_check (⇄ clarify_ask) → srd_review
          → trd (subgraph: draft_trd → trd_review)
          → requirements_first: handoff (next phase: design)
            design_first:       consistency_check → handoff (next phase: testing)

Every rejection, from any gate, goes through classify_rejection, which picks the
target (mode-aware) and escalates to the user once a gate's cap is exceeded.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .llm import PlanningLLM
from .nodes import Settings, make_nodes
from .state import PlanningState


def _route_clarify(state: PlanningState) -> str:
    if state.get("open_questions"):
        return "clarify_ask"
    if state.get("return_to"):
        return state["return_to"]
    return "scope_draft" if state.get("mode") == "requirements_first" else "design"


def _route_review(gate: str, on_approve: str):
    def route(state: PlanningState) -> str:
        return on_approve if (state.get("approved") or {}).get(gate) else "classify_rejection"
    return route


def _route_testability(state: PlanningState) -> str:
    return "clarify_ask" if state.get("open_questions") else "srd_review"


def _route_classify(state: PlanningState) -> str:
    return "escalate" if state.get("needs_escalation") else state["pending_route"]


def _route_escalate(state: PlanningState) -> str:
    return END if state.get("status") == "aborted" else state["pending_route"]


def _route_trd(state: PlanningState) -> str:
    outcome = state.get("trd_outcome")
    if outcome == "approved":
        return "consistency_check" if state.get("mode") == "design_first" else "handoff"
    if outcome == "untestable":
        return "classify_rejection"
    return "escalate"


def _route_design(state: PlanningState) -> str:
    return "draft_srd" if state.get("design_outcome") == "approved" else "escalate"


def _route_consistency(state: PlanningState) -> str:
    return "classify_rejection" if state.get("last_rejection") else "handoff"


def build_trd_subgraph(n: dict):
    g = StateGraph(PlanningState)
    g.add_node("draft_trd", n["draft_trd"])
    g.add_node("trd_review", n["trd_review"])
    g.add_edge(START, "draft_trd")
    g.add_conditional_edges("draft_trd", lambda s: END if s.get("trd_outcome") == "untestable" else "trd_review",
                            ["trd_review", END])
    g.add_conditional_edges("trd_review", lambda s: "draft_trd" if s.get("trd_outcome") == "rejected" else END,
                            ["draft_trd", END])
    return g.compile()   # no checkpointer here: it inherits the parent's


def build_design_subgraph(n: dict):
    """Part 2 placeholder. Replace with the real design/architecture subgraph later;
    keep the contract: reads clarify answers (and the SRD in requirements-first mode),
    writes state['design'] and state['design_outcome']."""
    g = StateGraph(PlanningState)
    g.add_node("design_draft", n["design_draft"])
    g.add_node("feasibility", n["feasibility"])
    g.add_node("design_review", n["design_review"])
    g.add_edge(START, "design_draft")
    g.add_edge("design_draft", "feasibility")
    g.add_edge("feasibility", "design_review")
    g.add_conditional_edges("design_review", lambda s: "design_draft" if s.get("design_outcome") == "rejected" else END,
                            ["design_draft", END])
    return g.compile()


def build_graph(llm: PlanningLLM, settings: Settings | None = None, checkpointer=None):
    n = make_nodes(llm, settings or Settings())
    g = StateGraph(PlanningState)

    for name in ("clarify_detect", "clarify_ask", "scope_draft", "feasibility", "scope_review",
                 "draft_srd", "testability_check", "srd_review", "classify_rejection",
                 "escalate", "consistency_check", "handoff"):
        g.add_node(name, n[name])
    g.add_node("trd", build_trd_subgraph(n))
    g.add_node("design", build_design_subgraph(n))

    g.add_edge(START, "clarify_detect")
    g.add_conditional_edges("clarify_detect", _route_clarify, ["clarify_ask", "scope_draft", "design", "draft_srd"])
    g.add_edge("clarify_ask", "clarify_detect")

    g.add_edge("scope_draft", "feasibility")
    g.add_edge("feasibility", "scope_review")
    g.add_conditional_edges("scope_review", _route_review("scope", "draft_srd"), ["draft_srd", "classify_rejection"])

    g.add_conditional_edges("design", _route_design, ["draft_srd", "escalate"])

    g.add_edge("draft_srd", "testability_check")
    g.add_conditional_edges("testability_check", _route_testability, ["clarify_ask", "srd_review"])
    g.add_conditional_edges("srd_review", _route_review("srd", "trd"), ["trd", "classify_rejection"])

    g.add_conditional_edges("trd", _route_trd, ["handoff", "consistency_check", "classify_rejection", "escalate"])
    g.add_conditional_edges("consistency_check", _route_consistency, ["classify_rejection", "handoff"])

    targets = ["scope_draft", "design", "draft_srd", "trd", "clarify_ask"]
    g.add_conditional_edges("classify_rejection", _route_classify, ["escalate", *targets])
    g.add_conditional_edges("escalate", _route_escalate, [END, *targets])

    g.add_edge("handoff", END)
    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


def initial_state(prompt: str, mode: str = "requirements_first") -> PlanningState:
    return {
        "mode": mode, "prompt": prompt, "answers": {}, "open_questions": [], "unresolved_questions": [],
        "clarify_rounds": 0, "return_to": None, "scope": None, "feasibility": None, "design": None,
        "srd": None, "trd": None, "changed_req_ids": [], "testability_issues": [], "last_rejection": None,
        "rejection_history": [], "iterations": {}, "pending_route": None, "needs_escalation": False,
        "approved": {}, "trd_outcome": None, "design_outcome": None, "status": "running",
        "next_phase": None, "artifact_paths": {},
    }
