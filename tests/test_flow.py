"""End-to-end route tests with a scripted fake LLM (no API key needed).

Each test drives the real compiled graph, pausing at every interrupt and resuming
with a scripted human answer, so every edge and gate is exercised for real.
"""
from __future__ import annotations

import os
import uuid

import pytest
from langgraph.types import Command

from planning import Settings, build_graph, initial_state
from planning.state import (
    SRD, TRD, ClarifyResult, Component, ConsistencyGap, ConsistencyResult, DesignDoc,
    FeasibilityReport, Question, RejectionClassification, Requirement, ScopeDoc, TestCase,
    TestabilityIssue, TestabilityResult,
)


# --------------------------------------------------------------------------- #
# fake LLM
# --------------------------------------------------------------------------- #
class FakeLLM:
    """Returns scripted outputs per schema. The last item of each queue repeats."""

    def __init__(self, script: dict[str, list]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str, str]] = []

    def structured(self, schema, system, user):
        name = schema.__name__
        self.calls.append((name, user))
        queue = self.script.get(name)
        if not queue:
            raise AssertionError(f"Unexpected LLM call for {name}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        return item(user) if callable(item) else item

    def inputs(self, name):
        return [u for n, u in self.calls if n == name]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
REQ1 = Requirement(id="REQ-001", statement="Users can request a password reset link by email.",
                   acceptance_criteria=["When a user submits a registered email, the system shall send a reset link within 60 seconds."])
REQ2 = Requirement(id="REQ-002", statement="Reset links expire.",
                   acceptance_criteria=["If a reset link is older than 60 minutes, then the system shall reject it with error LINK_EXPIRED."])
REQ2_V2 = Requirement(id="REQ-002", statement="Reset links expire.",
                      acceptance_criteria=["If a reset link is older than 30 minutes, then the system shall reject it with error LINK_EXPIRED."])
REQ2_VAGUE = Requirement(id="REQ-002", statement="Resets are fast.",
                         acceptance_criteria=["The system shall reset passwords quickly."])
REQ1_TAMPERED = Requirement(id="REQ-001", statement="SHOULD NOT APPEAR", acceptance_criteria=["The system shall do something else."])

SCOPE = ScopeDoc(summary="Password reset", goals=["Let users regain access"], in_scope=["Email reset"],
                 out_of_scope=["SMS"], constraints=[])
FEAS = FeasibilityReport(verdict="feasible", notes_md="Standard token flow.", srd_constraints=[])
TC1 = TestCase(id="TC-001a", req_id="REQ-001", criterion=REQ1.acceptance_criteria[0],
               steps=["Submit registered email"], expected="Link sent within 60 s")
TC2 = TestCase(id="TC-002a", req_id="REQ-002", criterion=REQ2.acceptance_criteria[0],
               steps=["Use a 61-minute-old link"], expected="LINK_EXPIRED")
TC2_V2 = TestCase(id="TC-002a", req_id="REQ-002", criterion=REQ2_V2.acceptance_criteria[0],
                  steps=["Use a 31-minute-old link"], expected="LINK_EXPIRED")
DESIGN = DesignDoc(summary="Token service", in_scope=["Email reset"], out_of_scope=["SMS"],
                   components=[Component(name="ResetService", responsibility="Issues tokens")])

APPROVE = {"action": "approve"}


def reject(text):
    return {"action": "reject", "feedback": text}


def drive(graph, state, answers, thread=None):
    """Run until the graph ends, feeding one scripted answer per interrupt."""
    config = {"configurable": {"thread_id": thread or str(uuid.uuid4())}, "recursion_limit": 200}
    answers = list(answers)
    payloads = []
    result = graph.invoke(state, config)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        payloads.append(payload)
        assert answers, f"Graph asked for more input than scripted: {payload}"
        result = graph.invoke(Command(resume=answers.pop(0)), config)
    assert not answers, f"Unused scripted answers: {answers}"
    return result, payloads


@pytest.fixture
def settings(tmp_path):
    return Settings(output_dir=str(tmp_path))


def base_script(**overrides):
    script = {
        "ClarifyResult": [ClarifyResult()],
        "ScopeDoc": [SCOPE],
        "FeasibilityReport": [FEAS],
        "SRD": [SRD(requirements=[REQ1, REQ2])],
        "TestabilityResult": [TestabilityResult()],
        "TRD": [TRD(test_cases=[TC1, TC2])],
    }
    script.update(overrides)
    return script


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_requirements_first_happy_path(settings):
    llm = FakeLLM(base_script(ClarifyResult=[
        ClarifyResult(questions=[Question(id="Q1", question="Email only or SMS too?", why_it_matters="Changes scope")]),
        ClarifyResult(),
    ]))
    g = build_graph(llm, settings)
    result, payloads = drive(g, initial_state("Build password reset"), [{"Q1": "Email only"}, APPROVE, APPROVE, APPROVE])

    assert [p.get("gate", p["type"]) for p in payloads] == ["clarify", "scope", "srd", "trd"]
    assert result["status"] == "handed_off" and result["next_phase"] == "design"
    for name in ("SRD.md", "TRD.md", "feasibility.md", "scope.md"):
        assert os.path.exists(result["artifact_paths"][name])
    assert "Email only" in llm.inputs("ScopeDoc")[0]          # answers reach the scope draft
    assert not llm.inputs("RejectionClassification")


def test_scope_rejection_revises_previous_draft_without_classifier_llm(settings):
    llm = FakeLLM(base_script())
    g = build_graph(llm, settings)
    result, payloads = drive(g, initial_state("Build password reset"),
                             [reject("Add SMS as a channel"), APPROVE, APPROVE, APPROVE])

    assert result["status"] == "handed_off"
    second = llm.inputs("ScopeDoc")[1]
    assert "PREVIOUS DRAFT" in second and "Add SMS as a channel" in second
    assert len(llm.inputs("FeasibilityReport")) == 2           # scope problem re-runs feasibility
    assert not llm.inputs("RejectionClassification")           # scope gate => scope kind, no LLM


def test_srd_detail_rejection_only_rewrites_affected_requirement(settings):
    llm = FakeLLM(base_script(
        SRD=[SRD(requirements=[REQ1, REQ2]), SRD(requirements=[REQ1_TAMPERED, REQ2_V2])],
        RejectionClassification=[RejectionClassification(kind="detail", affected_req_ids=["REQ-002"], rationale="one AC")],
        TRD=[TRD(test_cases=[TC1, TC2_V2])],
    ))
    g = build_graph(llm, settings)
    result, payloads = drive(g, initial_state("Build password reset"),
                             [APPROVE, reject("REQ-002 should expire after 30 minutes"), APPROVE, APPROVE])

    reqs = {r["id"]: r for r in result["srd"]["requirements"]}
    assert reqs["REQ-001"] == REQ1.model_dump()               # untouched even though the LLM changed it
    assert reqs["REQ-002"] == REQ2_V2.model_dump()
    assert result["changed_req_ids"] == ["REQ-002"]
    assert len(llm.inputs("FeasibilityReport")) == 1           # detail problem skips scope + feasibility
    assert "REQ-001" not in llm.inputs("TestabilityResult")[1]  # re-check only the changed requirement
    assert payloads[2]["gate"] == "srd" and "(changed)" in payloads[2]["artifact"]


def test_vague_requirement_becomes_a_question_then_redraft(settings):
    llm = FakeLLM(base_script(SRD=[SRD(requirements=[REQ1, REQ2_VAGUE]), SRD(requirements=[REQ1, REQ2])]))
    g = build_graph(llm, settings)
    result, payloads = drive(g, initial_state("Build password reset"),
                             [APPROVE, {"T-REQ-002": "within 2 seconds"}, APPROVE, APPROVE])

    assert payloads[1]["type"] == "clarify" and payloads[1]["questions"][0]["req_id"] == "REQ-002"
    redraft = llm.inputs("SRD")[1]
    assert "within 2 seconds" in redraft and "AFFECTED IDS: REQ-002" in redraft
    assert "REQ-002" not in llm.inputs("TestabilityResult")[0]  # caught deterministically, no tokens spent
    assert result["status"] == "handed_off"


def test_trd_backchannel_reopens_one_requirement_and_regenerates_only_its_tests(settings):
    untestable = TestabilityIssue(req_id="REQ-002", reason="No time limit given", question="How long is a link valid?")
    llm = FakeLLM(base_script(
        SRD=[SRD(requirements=[REQ1, REQ2]), SRD(requirements=[REQ1, REQ2_V2])],
        TRD=[TRD(test_cases=[TC1], untestable=[untestable]), TRD(test_cases=[TC2_V2])],
    ))
    g = build_graph(llm, settings)
    result, payloads = drive(g, initial_state("Build password reset"),
                             [APPROVE, APPROVE, {"T-REQ-002": "30 minutes"}, APPROVE, APPROVE])

    assert [p.get("gate", p["type"]) for p in payloads] == ["scope", "srd", "clarify", "srd", "trd"]
    assert {tc["id"] for tc in result["trd"]["test_cases"]} == {"TC-001a", "TC-002a"}
    assert "REQ-001" not in llm.inputs("TRD")[1]               # only the changed requirement went back to the LLM
    assert len(llm.inputs("ClarifyResult")) == 1               # no extra ambiguity pass on the return trip


def test_escalation_after_cap_then_abort(tmp_path):
    llm = FakeLLM(base_script())
    g = build_graph(llm, Settings(max_rejections=2, output_dir=str(tmp_path)))
    result, payloads = drive(g, initial_state("Build password reset"),
                             [reject("no"), reject("still no"), reject("no again"), {"action": "abort"}])

    assert payloads[-1]["type"] == "escalation" and payloads[-1]["gate"] == "scope"
    assert result["status"] == "aborted"
    assert len(llm.inputs("ScopeDoc")) == 3


def test_escalation_continue_with_guidance_resets_counter(tmp_path):
    llm = FakeLLM(base_script())
    g = build_graph(llm, Settings(max_rejections=1, output_dir=str(tmp_path)))
    result, _ = drive(g, initial_state("Build password reset"),
                      [reject("a"), reject("b"), {"action": "continue", "guidance": "Focus on email only"},
                       APPROVE, APPROVE, APPROVE])

    assert result["status"] == "handed_off"
    assert "Focus on email only" in llm.inputs("ScopeDoc")[-1]


def test_invalid_resume_is_reasked(settings):
    llm = FakeLLM(base_script())
    g = build_graph(llm, settings)
    result, payloads = drive(g, initial_state("Build password reset"),
                             [{"action": "reject"}, APPROVE, APPROVE, APPROVE])
    assert "error" in payloads[1]
    assert result["status"] == "handed_off"


def test_design_first_with_consistency_gap(settings):
    llm = FakeLLM(base_script(
        DesignDoc=[DESIGN, DESIGN.model_copy(update={"summary": "Token service with expiry job"})],
        ConsistencyResult=[ConsistencyResult(gaps=[ConsistencyGap(req_id="REQ-002", problem="No component expires links")]),
                           ConsistencyResult()],
        TRD=[TRD(test_cases=[TC1, TC2])],
    ))
    g = build_graph(llm, settings)
    result, payloads = drive(g, initial_state("Build password reset", mode="design_first"),
                             [APPROVE, APPROVE, APPROVE, APPROVE, APPROVE, APPROVE])

    assert [p["gate"] for p in payloads] == ["design", "srd", "trd", "design", "srd", "trd"]
    assert result["status"] == "handed_off" and result["next_phase"] == "testing"
    assert not llm.inputs("ScopeDoc")                          # no scope node in design-first
    assert "No component expires links" in llm.inputs("DesignDoc")[1]
    assert os.path.exists(result["artifact_paths"]["design.md"])
