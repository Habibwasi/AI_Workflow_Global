"""Markdown rendering for review payloads and the files written at handoff."""
from __future__ import annotations

import json
from typing import Optional


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) if items else "- (none)"


def scope_md(scope: dict) -> str:
    return (
        f"# Scope and goals\n\n{scope['summary']}\n\n"
        f"## Goals\n{_bullets(scope['goals'])}\n\n"
        f"## In scope\n{_bullets(scope['in_scope'])}\n\n"
        f"## Out of scope\n{_bullets(scope['out_of_scope'])}\n\n"
        f"## Fixed constraints\n{_bullets(scope.get('constraints', []))}\n"
    )


def feasibility_md(f: dict) -> str:
    return (
        f"# Feasibility\n\n**Verdict:** {f['verdict']}\n\n{f['notes_md']}\n\n"
        f"## Constraints carried into the SRD\n{_bullets(f.get('srd_constraints', []))}\n"
    )


def srd_md(srd: dict, changed: Optional[list[str]] = None) -> str:
    changed = set(changed or [])
    out = ["# Software Requirements Document\n"]
    for r in srd["requirements"]:
        mark = "  *(changed)*" if r["id"] in changed else ""
        out.append(f"## {r['id']} [{r['priority']}]{mark}\n{r['statement']}\n")
        out.append("**Acceptance criteria**\n" + _bullets(r["acceptance_criteria"]))
        if r.get("rationale"):
            out.append(f"\n_Rationale:_ {r['rationale']}")
        out.append("")
    return "\n".join(out)


def trd_md(trd: dict) -> str:
    out = ["# Test Requirements Document (acceptance level)\n"]
    if trd.get("environment"):
        out.append("## Environment\n" + _bullets(trd["environment"]) + "\n")
    out.append("## Traceability\n| Test | Requirement | Criterion |\n|---|---|---|")
    for tc in trd["test_cases"]:
        out.append(f"| {tc['id']} | {tc['req_id']} | {tc['criterion']} |")
    out.append("")
    for tc in trd["test_cases"]:
        out.append(f"### {tc['id']} → {tc['req_id']}")
        if tc.get("preconditions"):
            out.append("Preconditions:\n" + _bullets(tc["preconditions"]))
        out.append("Steps:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(tc["steps"], 1)))
        out.append(f"Expected: {tc['expected']}")
        if tc.get("test_data"):
            out.append("Test data:\n" + _bullets(tc["test_data"]))
        out.append("")
    return "\n".join(out)


def design_md(d: dict) -> str:
    comps = "\n".join(f"- **{c['name']}**: {c['responsibility']}" for c in d["components"])
    return (
        f"# Design\n\n{d['summary']}\n\n"
        f"## In scope\n{_bullets(d['in_scope'])}\n\n"
        f"## Out of scope\n{_bullets(d['out_of_scope'])}\n\n"
        f"## Components\n{comps}\n\n## Decisions\n{_bullets(d.get('decisions', []))}\n"
    )


def as_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
