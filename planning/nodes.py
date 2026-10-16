"""Node implementations for Part 1.

Two rules from the LangGraph interrupt docs shape this file:
* Review/question nodes contain ONLY ``interrupt()`` + validation. A node re-runs
  from the top on every resume, so an LLM call placed before ``interrupt()`` would
  run (and bill) again each time the user answers.
* Every loop has an explicit counter in state, so no loop relies on LangGraph's
  recursion limit to stop.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from . import prompts, render
from .audit import NAMES, Audit, changelog_md, decisions_jsonl, reason_for
from .llm import PlanningLLM
from .persistence import check_repo, commit_specs
from .state import (
    SRD,
    TRD,
    ClarifyResult,
    ConsistencyResult,
    DesignDoc,
    FeasibilityReport,
    PlanningState,
    Question,
    Rejection,
    RejectionClassification,
    ScopeDoc,
    TestabilityIssue,
    TestabilityResult,
)


@dataclass
class Settings:
    max_rejections: int = 3          # per gate, before escalating to you
    max_clarify_rounds: int = 4      # question rounds before moving on (open items shown at next gate)
    max_testability_rounds: int = 2  # self-check loops before showing leftover issues at SRD review
    output_dir: str = "planning_output"
    specs_repo: str | None = None    # path to the project's git repo; set it to commit specs on every approval
    specs_dir: str = "specs"         # folder inside specs_repo for the committed specs


# which files each approval commits (plus the decision log)
GATE_FILES = {
    "scope": lambda st: {"scope.md": render.scope_md(st["scope"]), "feasibility.md": render.feasibility_md(st["feasibility"])},
    "design": lambda st: {"design.md": render.design_md(st["design"]), "feasibility.md": render.feasibility_md(st["feasibility"])},
    "srd": lambda st: {"SRD.md": render.srd_md(st["srd"]), "srd.json": render.as_json(st["srd"])},
    "trd": lambda st: {"TRD.md": render.trd_md(st["trd"]), "trd.json": render.as_json(st["trd"])},
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
EARS_RE = re.compile(r"^\s*(the|when|while|if|where)\b.*\bshall\b", re.I | re.S)
VAGUE_RE = re.compile(
    r"\b(fast|quick(ly)?|user[- ]friendly|easy|easily|intuitive|efficient(ly)?|robust|"
    r"seamless(ly)?|appropriate(ly)?|as soon as possible|reasonable|etc\.?|and/or)\b",
    re.I,
)


def _bump(state: PlanningState, key: str) -> tuple[dict, int]:
    it = dict(state.get("iterations") or {})
    it[key] = it.get(key, 0) + 1
    return it, it[key]


def _history(state: PlanningState, rej: dict) -> list[dict]:
    return [*(state.get("rejection_history") or []), rej]


def _approved(state: PlanningState, gate: str, value: bool) -> dict:
    a = dict(state.get("approved") or {})
    a[gate] = value
    return a


def _context(state: PlanningState) -> str:
    parts = [f"MODE: {state.get('mode')}", f"REQUEST:\n{state['prompt']}"]
    answers = state.get("answers") or {}
    if answers:
        parts.append("ANSWERS FROM THE USER:\n" + "\n".join(f"- Q: {q}\n  A: {a}" for q, a in answers.items()))
    history = state.get("rejection_history") or []
    if history:
        parts.append("PAST REVIEWER FEEDBACK:\n" + "\n".join(f"- [{h['gate']}] {h['feedback']}" for h in history[-6:]))
    return "\n\n".join(parts)


def _review(gate: str, payload: dict) -> dict:
    """Interrupt for approve/reject and keep asking until the answer is valid."""
    base = {"type": "review", "gate": gate, "actions": ["approve", "reject"], **payload}
    msg = base
    while True:
        resp = interrupt(msg)
        if isinstance(resp, dict):
            if resp.get("action") == "approve":
                return resp
            if resp.get("action") == "reject" and str(resp.get("feedback", "")).strip():
                return resp
        msg = {**base, "error": "Resume with {'action': 'approve'} or {'action': 'reject', 'feedback': '<why>'}"}


def deterministic_testability(srd: dict, only: set[str] | None = None) -> list[TestabilityIssue]:
    """Cheap checks before spending tokens: EARS shape and vague words."""
    issues = []
    for r in srd["requirements"]:
        if only is not None and r["id"] not in only:
            continue
        crit = r.get("acceptance_criteria") or []
        if not crit:
            issues.append(TestabilityIssue(req_id=r["id"], reason="No acceptance criteria",
                                           question=f"How should we judge that {r['id']} ('{r['statement']}') works?"))
            continue
        for c in crit:
            if not EARS_RE.match(c):
                issues.append(TestabilityIssue(req_id=r["id"], reason=f"Not in EARS form: '{c}'",
                                               question=f"For {r['id']}, what exact trigger and expected result should '{c}' mean?"))
                break
            m = VAGUE_RE.search(c)
            if m:
                issues.append(TestabilityIssue(req_id=r["id"], reason=f"Vague term '{m.group(0)}' in: '{c}'",
                                               question=f"For {r['id']}, what measurable value should replace '{m.group(0)}'?"))
                break
    return issues


def merge_srd(prev: dict, new: dict, affected: list[str]) -> dict:
    """Keep unaffected requirements byte-for-byte; take affected/new ones from the redraft."""
    affected_set = set(affected)
    new_by_id = {r["id"]: r for r in new["requirements"]}
    merged = []
    for r in prev["requirements"]:
        if r["id"] in affected_set:
            if r["id"] in new_by_id:          # dropped if the redraft removed it
                merged.append(new_by_id[r["id"]])
        else:
            merged.append(r)
    prev_ids = {r["id"] for r in prev["requirements"]}
    merged += [r for r in new["requirements"] if r["id"] not in prev_ids]
    return {"requirements": merged}


def changed_ids(prev: dict | None, new: dict) -> list[str]:
    if not prev:
        return [r["id"] for r in new["requirements"]]
    old = {r["id"]: r for r in prev["requirements"]}
    return [r["id"] for r in new["requirements"] if old.get(r["id"]) != r]


# --------------------------------------------------------------------------- #
# node factory
# --------------------------------------------------------------------------- #
def make_nodes(llm: PlanningLLM, s: Settings) -> dict[str, Callable[..., dict[str, Any]]]:
    if s.specs_repo:
        check_repo(s.specs_repo)

    def _thread(config: RunnableConfig) -> str:
        return str((config.get("configurable") or {}).get("thread_id", "run"))

    def _recorded_review(state: PlanningState, config: RunnableConfig, gate: str, resp: dict) -> dict:
        """Log the decision; on approval, commit the approved artifact with its reasons."""
        a = Audit(state)
        latest = a.latest(gate)
        v = latest["version"] if latest else 0
        a.event("review", gate=gate, artifact=NAMES[gate], version=v,
                decision=resp["action"], feedback=resp.get("feedback"))
        if resp["action"] == "approve" and s.specs_repo:
            files = GATE_FILES[gate](state)
            files["decisions.jsonl"] = decisions_jsonl(a.log)
            files["CHANGELOG.md"] = changelog_md(a.log)
            rejects = [e for e in a.log if e["event"] == "review" and e["gate"] == gate and e["decision"] == "reject"]
            subject = f"specs: {NAMES[gate]} v{v} approved"
            body = [f"Why this version: {latest['reason'] if latest else 'n/a'}"]
            if latest and latest.get("changed"):
                body.append(f"Changed: {', '.join(latest['changed'])}")
            body.append(f"Rejections at this gate: {len(rejects)}")
            body += [f"  - v{e['version']} rejected: {e['feedback']}" for e in rejects]
            body.append(f"Planning run: {_thread(config)}")
            sha = commit_specs(s.specs_repo, s.specs_dir, files, subject, "\n".join(body))
            if sha:
                a.event("committed", gate=gate, sha=sha, subject=subject)
        return a.update()

    # ---------------- clarify (shared by both modes) ----------------
    def clarify_detect(state: PlanningState) -> dict:
        # Returning from a testability / TRD question round: the answers are already
        # targeted, so go straight back without another full ambiguity pass.
        if state.get("return_to"):
            return {"open_questions": []}
        res = llm.structured(ClarifyResult, prompts.CLARIFY, _context(state))
        answered = set((state.get("answers") or {}).keys())
        qs = [q.model_dump() for q in res.questions if q.question not in answered]
        if state.get("clarify_rounds", 0) >= s.max_clarify_rounds:
            return {"open_questions": [], "unresolved_questions": qs}
        return {"open_questions": qs, "unresolved_questions": []}

    def clarify_ask(state: PlanningState) -> dict:
        qs = state.get("open_questions") or []
        payload = {
            "type": "clarify",
            "questions": qs,
            "instructions": "Resume with {question_id: answer} for every question.",
        }
        msg = payload
        while True:
            resp = interrupt(msg)
            if isinstance(resp, dict) and all(str(resp.get(q["id"], "")).strip() for q in qs):
                break
            missing = [q["id"] for q in qs if not (isinstance(resp, dict) and str(resp.get(q["id"], "")).strip())]
            msg = {**payload, "error": f"Missing answers for: {', '.join(missing)}"}
        answers = dict(state.get("answers") or {})
        for q in qs:
            answers[q["question"]] = str(resp[q["id"]]).strip()
        a = Audit(state)
        a.event("clarified", answers=[{
            "id": q["id"], "question": q["question"], "answer": answers[q["question"]],
            "source": f"requirement check ({q['req_id']})" if q.get("req_id") else "clarify",
        } for q in qs])
        update: dict = {"answers": answers, "open_questions": [], **a.update()}
        if not state.get("return_to"):
            update["clarify_rounds"] = state.get("clarify_rounds", 0) + 1
        return update

    # ---------------- requirements-first: scope + feasibility ----------------
    def scope_draft(state: PlanningState) -> dict:
        user = _context(state)
        rej = state.get("last_rejection")
        if state.get("scope") and rej and rej.get("kind") == "scope":
            user += f"\n\nPREVIOUS DRAFT:\n{render.as_json(state['scope'])}\n\nREJECTION FEEDBACK:\n{rej['feedback']}"
        scope = llm.structured(ScopeDoc, prompts.SCOPE, user).model_dump()
        a = Audit(state)
        a.version("scope", scope, render.scope_md(scope), reason_for(state))
        return {"scope": scope, **a.update()}

    def feasibility(state: PlanningState) -> dict:
        if state.get("mode") == "design_first":
            subject, basis = render.design_md(state["design"]), "design"
        else:
            subject, basis = render.scope_md(state["scope"]), "scope"
        rep = llm.structured(FeasibilityReport, prompts.FEASIBILITY, f"{_context(state)}\n\nSUBJECT:\n{subject}").model_dump()
        a = Audit(state)
        latest = a.latest(basis)
        a.version("feasibility", rep, render.feasibility_md(rep),
                  f"assessed {NAMES[basis]} v{latest['version'] if latest else '?'} (verdict: {rep['verdict']})")
        return {"feasibility": rep, **a.update()}

    def scope_review(state: PlanningState, config: RunnableConfig) -> dict:
        resp = _review("scope", {
            "artifact": render.scope_md(state["scope"]),
            "feasibility": render.feasibility_md(state["feasibility"]),
            "unresolved_questions": state.get("unresolved_questions") or [],
        })
        audit = _recorded_review(state, config, "scope", resp)
        if resp["action"] == "approve":
            return {"approved": _approved(state, "scope", True), "last_rejection": None, **audit}
        rej = Rejection(gate="scope", feedback=resp["feedback"]).model_dump()
        return {"approved": _approved(state, "scope", False), "last_rejection": rej, **audit}

    # ---------------- SRD ----------------
    def draft_srd(state: PlanningState) -> dict:
        prev = state.get("srd")
        rej = state.get("last_rejection")
        affected = rej.get("affected_req_ids", []) if rej and rej.get("kind") == "detail" else []

        if state.get("mode") == "design_first":
            basis = render.design_md(state["design"])
        else:
            basis = render.scope_md(state["scope"])
        user = (
            f"{_context(state)}\n\nBASIS (scope or design):\n{basis}\n\n"
            f"FEASIBILITY CONSTRAINTS:\n{render.as_json(state['feasibility'].get('srd_constraints', []))}"
        )
        if prev:
            user += f"\n\nPREVIOUS SRD:\n{render.as_json(prev)}"
        if rej and rej.get("kind") == "detail":
            user += f"\n\nFEEDBACK TO ADDRESS:\n{rej['feedback']}\n\nAFFECTED IDS: {', '.join(affected) or '(see feedback)'}"

        new = llm.structured(SRD, prompts.SRD, user).model_dump()
        srd = merge_srd(prev, new, affected) if (prev and affected) else new
        changed = changed_ids(prev, srd)
        a = Audit(state)
        a.version("srd", srd, render.srd_md(srd), reason_for(state), changed)
        return {"srd": srd, "changed_req_ids": changed, "return_to": None, **a.update()}

    def testability_check(state: PlanningState) -> dict:
        srd = state["srd"]
        changed = set(state.get("changed_req_ids") or [])
        issues = deterministic_testability(srd, only=changed)
        flagged = {i.req_id for i in issues}
        to_llm = [r for r in srd["requirements"] if r["id"] in changed and r["id"] not in flagged]
        if to_llm:
            res = llm.structured(TestabilityResult, prompts.TESTABILITY, render.as_json({"requirements": to_llm}))
            issues += [i for i in res.issues if i.req_id not in flagged]

        if not issues:
            return {"testability_issues": []}

        a = Audit(state)
        a.event("self_check", check="testability", issues=[i.model_dump() for i in issues])
        iters, n = _bump(state, "testability")
        if n > s.max_testability_rounds:
            # Stop looping; show what is left to the user at SRD review instead.
            return {"testability_issues": [i.model_dump() for i in issues], "iterations": iters, **a.update()}

        questions = [Question(id=f"T-{i.req_id}", question=i.question, why_it_matters=i.reason,
                              req_id=i.req_id).model_dump() for i in issues]
        rej = Rejection(gate="testability", kind="detail", feedback="; ".join(i.reason for i in issues),
                        affected_req_ids=sorted({i.req_id for i in issues})).model_dump()
        return {"open_questions": questions, "return_to": "draft_srd", "last_rejection": rej,
                "iterations": iters, "testability_issues": [], **a.update()}

    def srd_review(state: PlanningState, config: RunnableConfig) -> dict:
        rej = state.get("last_rejection")
        resp = _review("srd", {
            "artifact": render.srd_md(state["srd"], state.get("changed_req_ids")),
            "addressing_feedback": rej["feedback"] if rej else None,
            "open_testability_issues": state.get("testability_issues") or [],
            "unresolved_questions": state.get("unresolved_questions") or [],
        })
        audit = _recorded_review(state, config, "srd", resp)
        if resp["action"] == "approve":
            iters = dict(state.get("iterations") or {})
            iters["testability"] = 0
            return {"approved": _approved(state, "srd", True), "last_rejection": None, "iterations": iters, **audit}
        return {"approved": _approved(state, "srd", False),
                "last_rejection": Rejection(gate="srd", feedback=resp["feedback"]).model_dump(), **audit}

    # ---------------- rejection routing ----------------
    def classify_rejection(state: PlanningState) -> dict:
        rej = dict(state["last_rejection"])
        design_first = state.get("mode") == "design_first"

        method, rationale = "preset by the step that raised it", None
        if rej.get("kind") is None:
            if rej["gate"] == "scope":
                rej["kind"] = "scope"                     # a scope-gate rejection is a scope problem by definition
                method = "rule (scope gate)"
            elif rej["gate"] == "srd":
                c = llm.structured(
                    RejectionClassification, prompts.CLASSIFY,
                    f"FEEDBACK:\n{rej['feedback']}\n\nSRD:\n{render.as_json(state['srd'])}",
                )
                rej["kind"], rej["affected_req_ids"] = c.kind, c.affected_req_ids
                method, rationale = "LLM classifier", c.rationale

        if state.get("open_questions"):
            target = "clarify_ask"                        # TRD back-channel: ask first, then redraft SRD
        elif rej["kind"] == "scope":
            target = "design" if design_first else "scope_draft"
        elif rej["kind"] == "design":
            target = "design"
        else:
            target = "draft_srd"

        iters, n = _bump(state, rej["gate"])
        escalate_now = n > s.max_rejections
        a = Audit(state)
        a.event("rejection_classified", gate=rej["gate"], kind=rej["kind"],
                affected_req_ids=rej.get("affected_req_ids", []), method=method, rationale=rationale,
                route="escalate to you" if escalate_now else target)
        return {
            "last_rejection": rej,
            "rejection_history": _history(state, rej),
            "iterations": iters,
            "pending_route": target,
            "needs_escalation": escalate_now,
            **a.update(),
        }

    def escalate(state: PlanningState) -> dict:
        rej = state["last_rejection"]
        gate = rej["gate"]
        payload = {
            "type": "escalation",
            "gate": gate,
            "rejections_at_gate": (state.get("iterations") or {}).get(gate, 0),
            "recent_feedback": [h["feedback"] for h in (state.get("rejection_history") or []) if h["gate"] == gate][-5:],
            "next_step_if_continue": state.get("pending_route"),
            "actions": ["continue", "abort"],
            "instructions": "Resume with {'action': 'continue', 'guidance': '<optional>'} or {'action': 'abort'}.",
        }
        msg = payload
        while True:
            resp = interrupt(msg)
            if isinstance(resp, dict) and resp.get("action") in ("continue", "abort"):
                break
            msg = {**payload, "error": "action must be 'continue' or 'abort'"}
        a = Audit(state)
        a.event("escalated", gate=gate, rejections=(state.get("iterations") or {}).get(gate, 0),
                decision=resp["action"], guidance=str(resp.get("guidance", "")).strip() or None)
        if resp["action"] == "abort":
            return {"status": "aborted", "needs_escalation": False, **a.update()}
        iters = dict(state.get("iterations") or {})
        iters[gate] = 0
        update: dict = {"needs_escalation": False, "iterations": iters, **a.update()}
        if str(resp.get("guidance", "")).strip():
            update["last_rejection"] = {**rej, "feedback": f"{rej['feedback']}\nGuidance: {resp['guidance']}"}
        return update

    # ---------------- TRD subgraph ----------------
    def draft_trd(state: PlanningState) -> dict:
        srd = state["srd"]
        prev = state.get("trd")
        rej = state.get("last_rejection")
        local_feedback = rej["feedback"] if rej and rej.get("gate") == "trd" and rej.get("kind") is None else None
        changed = set(state.get("changed_req_ids") or [])
        req_ids = {r["id"] for r in srd["requirements"]}

        if prev and not local_feedback and changed and changed != req_ids:
            # Only regenerate tests for requirements that changed.
            subset = {"requirements": [r for r in srd["requirements"] if r["id"] in changed]}
            new = llm.structured(TRD, prompts.TRD, f"SRD (changed requirements only):\n{render.as_json(subset)}").model_dump()
            kept = [tc for tc in prev["test_cases"] if tc["req_id"] not in changed and tc["req_id"] in req_ids]
            trd = {"test_cases": kept + new["test_cases"],
                   "environment": sorted(set(prev.get("environment", [])) | set(new.get("environment", []))),
                   "untestable": new.get("untestable", [])}
        else:
            user = f"SRD:\n{render.as_json(srd)}"
            if prev:
                user += f"\n\nPREVIOUS TRD:\n{render.as_json(prev)}"
            if local_feedback:
                user += f"\n\nREVIEW FEEDBACK:\n{local_feedback}"
            trd = llm.structured(TRD, prompts.TRD, user).model_dump()

        trd["test_cases"] = [tc for tc in trd["test_cases"] if tc["req_id"] in req_ids]   # drop orphans

        old_tcs = {tc["id"]: tc for tc in (prev or {}).get("test_cases", [])}
        changed_tcs = [tc["id"] for tc in trd["test_cases"] if old_tcs.get(tc["id"]) != tc]
        a = Audit(state)
        srd_v = (a.latest("srd") or {}).get("version", "?")
        a.version("trd", trd, render.trd_md(trd),
                  reason_for(state, f"derived from SRD v{srd_v}") if local_feedback else f"derived from SRD v{srd_v}",
                  changed_tcs)

        if trd.get("untestable"):
            issues = trd["untestable"]
            a.event("self_check", check="TRD drafting (untestable requirements)", issues=issues)
            questions = [Question(id=f"T-{i['req_id']}", question=i["question"], why_it_matters=i["reason"],
                                  req_id=i["req_id"]).model_dump() for i in issues]
            rej = Rejection(gate="trd", kind="detail", feedback="Untestable: " + "; ".join(i["reason"] for i in issues),
                            affected_req_ids=sorted({i["req_id"] for i in issues})).model_dump()
            return {"trd": trd, "trd_outcome": "untestable", "last_rejection": rej,
                    "open_questions": questions, "return_to": "draft_srd", **a.update()}
        return {"trd": trd, "trd_outcome": None, **a.update()}

    def trd_review(state: PlanningState, config: RunnableConfig) -> dict:
        trd = state["trd"]
        covered = {tc["req_id"] for tc in trd["test_cases"]}
        gaps = [r["id"] for r in state["srd"]["requirements"] if r["id"] not in covered]
        resp = _review("trd", {"artifact": render.trd_md(trd), "requirements_without_tests": gaps})
        audit = _recorded_review(state, config, "trd", resp)
        if resp["action"] == "approve":
            return {"trd_outcome": "approved", "approved": _approved(state, "trd", True), "last_rejection": None, **audit}
        rej = Rejection(gate="trd", feedback=resp["feedback"]).model_dump()
        iters, n = _bump(state, "trd")
        update = {"approved": _approved(state, "trd", False), "last_rejection": rej,
                  "rejection_history": _history(state, rej), "iterations": iters, **audit}
        if n > s.max_rejections:
            return {**update, "trd_outcome": "escalate", "needs_escalation": True, "pending_route": "trd"}
        return {**update, "trd_outcome": "rejected"}

    # ---------------- design (Part 2 placeholder) ----------------
    def design_draft(state: PlanningState) -> dict:
        user = _context(state)
        rej = state.get("last_rejection")
        if state.get("design") and rej:
            user += f"\n\nPREVIOUS DESIGN:\n{render.as_json(state['design'])}\n\nFEEDBACK:\n{rej['feedback']}"
        d = llm.structured(DesignDoc, prompts.DESIGN, user).model_dump()
        a = Audit(state)
        a.version("design", d, render.design_md(d), reason_for(state))
        return {"design": d, "design_outcome": None, **a.update()}

    def design_review(state: PlanningState, config: RunnableConfig) -> dict:
        resp = _review("design", {
            "artifact": render.design_md(state["design"]),
            "feasibility": render.feasibility_md(state["feasibility"]),
            "unresolved_questions": state.get("unresolved_questions") or [],
        })
        audit = _recorded_review(state, config, "design", resp)
        if resp["action"] == "approve":
            return {"design_outcome": "approved", "approved": _approved(state, "design", True),
                    "last_rejection": None, **audit}
        rej = Rejection(gate="design", kind="design", feedback=resp["feedback"]).model_dump()
        iters, n = _bump(state, "design")
        update = {"last_rejection": rej, "rejection_history": _history(state, rej), "iterations": iters, **audit}
        if n > s.max_rejections:
            return {**update, "design_outcome": "escalate", "needs_escalation": True, "pending_route": "design"}
        return {**update, "design_outcome": "rejected"}

    # ---------------- design-first only: cross-artifact check ----------------
    def consistency_check(state: PlanningState) -> dict:
        res = llm.structured(
            ConsistencyResult, prompts.CONSISTENCY,
            f"DESIGN:\n{render.as_json(state['design'])}\n\nSRD:\n{render.as_json(state['srd'])}",
        )
        a = Audit(state)
        a.event("consistency_check", gaps=[g.model_dump() for g in res.gaps])
        if not res.gaps:
            return {"last_rejection": None, **a.update()}
        rej = Rejection(gate="consistency", kind="design",
                        feedback="Design gaps: " + "; ".join(f"{g.req_id}: {g.problem}" for g in res.gaps),
                        affected_req_ids=[g.req_id for g in res.gaps]).model_dump()
        return {"last_rejection": rej, **a.update()}

    # ---------------- handoff ----------------
    def handoff(state: PlanningState, config: RunnableConfig) -> dict:
        thread = (config.get("configurable") or {}).get("thread_id", "run")
        out = os.path.join(s.output_dir, str(thread))
        os.makedirs(out, exist_ok=True)
        files = {
            "SRD.md": render.srd_md(state["srd"]),
            "TRD.md": render.trd_md(state["trd"]),
            "feasibility.md": render.feasibility_md(state["feasibility"]),
            "srd.json": render.as_json(state["srd"]),
            "trd.json": render.as_json(state["trd"]),
        }
        if state.get("scope"):
            files["scope.md"] = render.scope_md(state["scope"])
        if state.get("design"):
            files["design.md"] = render.design_md(state["design"])

        nxt = "testing" if state.get("mode") == "design_first" else "design"
        a = Audit(state)
        a.event("handoff", next_phase=nxt)
        files["decisions.jsonl"] = decisions_jsonl(a.log)
        files["CHANGELOG.md"] = changelog_md(a.log)
        # every version of every artifact, with the reason and the diff to the previous one
        for artifact, versions in a.versions.items():
            for v in versions:
                stem = f"versions/{NAMES[artifact]}_v{v['version']}"
                files[f"{stem}.md"] = f"<!-- Why: {v['reason']} -->\n\n{v['markdown']}"
                if v["diff"]:
                    files[f"{stem}.diff"] = v["diff"]

        paths = {}
        for name, text in files.items():
            p = os.path.join(out, name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            paths[name] = p

        if s.specs_repo:
            final = {"decisions.jsonl": files["decisions.jsonl"], "CHANGELOG.md": files["CHANGELOG.md"]}
            sha = commit_specs(s.specs_repo, s.specs_dir, final, "specs: planning complete",
                               f"Next phase: {nxt}\nPlanning run: {thread}")
            if sha:
                a.event("committed", gate="handoff", sha=sha, subject="specs: planning complete")

        return {"status": "handed_off", "next_phase": nxt, "artifact_paths": paths, **a.update()}

    return {
        "clarify_detect": clarify_detect,
        "clarify_ask": clarify_ask,
        "scope_draft": scope_draft,
        "feasibility": feasibility,
        "scope_review": scope_review,
        "draft_srd": draft_srd,
        "testability_check": testability_check,
        "srd_review": srd_review,
        "classify_rejection": classify_rejection,
        "escalate": escalate,
        "draft_trd": draft_trd,
        "trd_review": trd_review,
        "design_draft": design_draft,
        "design_review": design_review,
        "consistency_check": consistency_check,
        "handoff": handoff,
    }
