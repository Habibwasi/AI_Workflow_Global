"""State and artifact schemas for Part 1 (planning).

Artifacts are Pydantic models so the LLM can return them as structured output.
They are stored in graph state as plain dicts (``model_dump()``) so any LangGraph
checkpointer can serialise them without custom type registration.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field

Mode = Literal["requirements_first", "design_first"]
Gate = Literal["scope", "srd", "trd", "design", "testability", "consistency"]
RejectionKind = Literal["scope", "detail", "design"]


# --------------------------------------------------------------------------- #
# Clarify
# --------------------------------------------------------------------------- #
class Question(BaseModel):
    id: str = Field(description="Short stable id, e.g. Q1")
    question: str
    why_it_matters: str = Field(description="What changes in scope/requirements depending on the answer")
    req_id: Optional[str] = Field(default=None, description="Set when the question is about one requirement")


class ClarifyResult(BaseModel):
    questions: list[Question] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Requirements-first: scope + feasibility
# --------------------------------------------------------------------------- #
class ScopeDoc(BaseModel):
    summary: str
    goals: list[str]
    in_scope: list[str]
    out_of_scope: list[str]
    constraints: list[str] = Field(default_factory=list, description="Fixed constraints the user stated")


class FeasibilityReport(BaseModel):
    verdict: Literal["feasible", "at_risk", "infeasible"]
    notes_md: str = Field(description="Technical findings on how it could be built: approach options, risks, effort. Written to feasibility.md, never into the SRD")
    srd_constraints: list[str] = Field(
        default_factory=list,
        description="Only conclusions that change WHAT is built, e.g. 'sync within 5 minutes' instead of 'real-time'",
    )


# --------------------------------------------------------------------------- #
# SRD
# --------------------------------------------------------------------------- #
class Requirement(BaseModel):
    id: str = Field(description="REQ-001, REQ-002, ... Stable across revisions")
    statement: str = Field(description="What the system must do. No technology choices")
    acceptance_criteria: list[str] = Field(description="EARS sentences, each pass/fail testable")
    priority: Literal["must", "should", "could"] = "must"
    rationale: str = ""


class SRD(BaseModel):
    requirements: list[Requirement]


class TestabilityIssue(BaseModel):
    req_id: str
    reason: str
    question: str = Field(description="One question to the user that would make the requirement testable")


class TestabilityResult(BaseModel):
    issues: list[TestabilityIssue] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #
class RejectionClassification(BaseModel):
    kind: Literal["scope", "detail"]
    affected_req_ids: list[str] = Field(default_factory=list)
    rationale: str


class Rejection(BaseModel):
    gate: Gate
    feedback: str
    kind: Optional[RejectionKind] = None
    affected_req_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# TRD
# --------------------------------------------------------------------------- #
class TestCase(BaseModel):
    id: str = Field(description="TC-001a, TC-001b ... prefix matches the requirement number")
    req_id: str
    criterion: str = Field(description="The EARS criterion this test verifies, copied verbatim")
    preconditions: list[str] = Field(default_factory=list)
    steps: list[str]
    expected: str
    test_data: list[str] = Field(default_factory=list)


class TRD(BaseModel):
    test_cases: list[TestCase]
    environment: list[str] = Field(default_factory=list)
    untestable: list[TestabilityIssue] = Field(
        default_factory=list,
        description="Requirements that cannot be tested as written. Never invent criteria for them",
    )


# --------------------------------------------------------------------------- #
# Design (Part 2 placeholder, needed for design-first mode)
# --------------------------------------------------------------------------- #
class Component(BaseModel):
    name: str
    responsibility: str


class DesignDoc(BaseModel):
    summary: str
    in_scope: list[str]
    out_of_scope: list[str]
    components: list[Component]
    decisions: list[str] = Field(default_factory=list)


class ConsistencyGap(BaseModel):
    req_id: str
    problem: str


class ConsistencyResult(BaseModel):
    gaps: list[ConsistencyGap] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #
class PlanningState(TypedDict, total=False):
    # input
    mode: Mode
    prompt: str

    # clarify
    answers: dict[str, str]            # question text -> answer
    open_questions: list[dict]         # Question dicts waiting for the user
    unresolved_questions: list[dict]   # left open when the clarify cap was hit; shown at the next gate
    clarify_rounds: int
    return_to: Optional[str]           # where clarify hands back to (None = mode router)

    # artifacts (dicts of the models above)
    scope: Optional[dict]
    feasibility: Optional[dict]
    design: Optional[dict]
    srd: Optional[dict]
    trd: Optional[dict]
    changed_req_ids: list[str]         # set by draft_srd; draft_trd regenerates only these
    testability_issues: list[dict]     # left over when the testability cap was hit

    # rejection handling
    last_rejection: Optional[dict]
    rejection_history: list[dict]      # managed by hand (no reducer, see README)
    iterations: dict[str, int]         # per-gate counters for the cap
    pending_route: Optional[str]       # where the classifier wants to go next
    needs_escalation: bool

    # gates / subgraph outcomes
    approved: dict[str, bool]
    trd_outcome: Optional[Literal["approved", "untestable", "escalate"]]
    design_outcome: Optional[Literal["approved", "escalate"]]

    # observability: once a run starts with --no-content, every resume keeps it
    hide_trace_content: bool

    # audit trail (see audit.py)
    audit_log: list[dict]                       # every decision, oldest first
    artifact_versions: dict[str, list[dict]]    # artifact -> [{version, reason, changed, content, markdown, diff}]

    # end
    status: Literal["running", "handed_off", "aborted"]
    next_phase: Optional[Literal["design", "testing"]]
    artifact_paths: dict[str, str]
