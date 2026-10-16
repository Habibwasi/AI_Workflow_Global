"""Audit trail: versioned artifacts and a decision log kept in graph state.

Why this exists: a checkpointer stores raw state snapshots, which answer "what was
the state at step N" but not "why did REQ-004 change". This module records the why:
every draft version (with the reason it was made and a diff), every review decision,
every answer, every classification and route.

State keys written here (overwrite semantics, no reducers, see README):
    audit_log:          list[event dict]
    artifact_versions:  {artifact_name: [version dict, ...]}
"""
from __future__ import annotations

import difflib
import json
from datetime import datetime, timezone
from typing import Optional


NAMES = {"scope": "Scope", "feasibility": "Feasibility", "srd": "SRD", "trd": "TRD",
         "design": "Design", "testability": "testability", "consistency": "consistency"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Audit:
    """Collects audit updates inside one node and returns them as a state update."""

    def __init__(self, state: dict):
        self.log = list(state.get("audit_log") or [])
        self.versions = {k: list(v) for k, v in (state.get("artifact_versions") or {}).items()}
        self._touched_versions = False

    def event(self, event: str, **data) -> dict:
        e = {"ts": now(), "event": event, **data}
        self.log.append(e)
        return e

    def latest(self, artifact: str) -> Optional[dict]:
        v = self.versions.get(artifact) or []
        return v[-1] if v else None

    def version(self, artifact: str, content: dict, markdown: str, reason: str,
                changed: Optional[list[str]] = None) -> int:
        """Record a new version unless the content is identical to the latest one."""
        prev = self.latest(artifact)
        if prev and prev["content"] == content:
            return prev["version"]
        n = (prev["version"] + 1) if prev else 1
        diff = ""
        if prev:
            diff = "".join(difflib.unified_diff(
                prev["markdown"].splitlines(keepends=True), markdown.splitlines(keepends=True),
                fromfile=f"{NAMES[artifact]} v{prev['version']}", tofile=f"{NAMES[artifact]} v{n}"))
        self.versions.setdefault(artifact, []).append({
            "version": n, "ts": now(), "reason": reason, "changed": changed or [],
            "content": content, "markdown": markdown, "diff": diff,
        })
        self._touched_versions = True
        self.event("artifact_drafted", artifact=NAMES[artifact], version=n, reason=reason, changed=changed or [])
        return n

    def update(self) -> dict:
        out = {"audit_log": self.log}
        if self._touched_versions:
            out["artifact_versions"] = self.versions
        return out


def reason_for(state: dict, default: str = "initial draft") -> str:
    """Explain why a draft is being (re)made, from the rejection that triggered it."""
    rej = state.get("last_rejection")
    if not rej:
        return default
    source = {
        "testability": "testability self-check",
        "consistency": "design/requirements consistency check",
        "trd": "TRD found requirements untestable" if rej.get("kind") == "detail" else "TRD review rejection",
    }.get(rej["gate"], f"{NAMES.get(rej['gate'], rej['gate'])} review rejection")
    return f"{source}: {rej['feedback']}"


# --------------------------------------------------------------------------- #
# Rendering for files
# --------------------------------------------------------------------------- #
def decisions_jsonl(log: list[dict]) -> str:
    return "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in log)


def _line(e: dict) -> str:
    t = e["ts"].replace("T", " ").replace("+00:00", " UTC")
    ev = e["event"]
    if ev == "run_started":
        return f"- **{t}** Run started ({e['mode']}). Request: {e['prompt']}"
    if ev == "clarified":
        qa = "\n".join(f"    - Q ({x['source']}): {x['question']}\n      A: {x['answer']}" for x in e["answers"])
        return f"- **{t}** You answered {len(e['answers'])} question(s):\n{qa}"
    if ev == "artifact_drafted":
        ch = f" Changed: {', '.join(e['changed'])}." if e.get("changed") else ""
        return f"- **{t}** {e['artifact']} v{e['version']} drafted. Why: {e['reason']}.{ch}"
    if ev == "review":
        if e["decision"] == "approve":
            return f"- **{t}** You **approved** {e['artifact']} v{e['version']}."
        return f"- **{t}** You **rejected** {e['artifact']} v{e['version']}: \"{e['feedback']}\""
    if ev == "rejection_classified":
        ids = f" (affects {', '.join(e['affected_req_ids'])})" if e.get("affected_req_ids") else ""
        return (f"- **{t}** Rejection at {NAMES.get(e['gate'], e['gate'])} classified as **{e['kind']}**{ids} "
                f"by {e['method']}; next step: {e['route']}.")
    if ev == "self_check":
        items = "; ".join(f"{i['req_id']}: {i['reason']}" for i in e["issues"])
        return f"- **{t}** {e['check']} check flagged: {items}"
    if ev == "consistency_check":
        if not e["gaps"]:
            return f"- **{t}** Design/requirements consistency check passed."
        return f"- **{t}** Consistency gaps: " + "; ".join(f"{g['req_id']}: {g['problem']}" for g in e["gaps"])
    if ev == "escalated":
        g = f" Guidance: {e['guidance']}" if e.get("guidance") else ""
        return f"- **{t}** Escalation at {e['gate']} after {e['rejections']} rejections: you chose **{e['decision']}**.{g}"
    if ev == "committed":
        return f"- **{t}** Committed `{e['sha'][:7]}`: {e['subject']}"
    if ev == "handoff":
        return f"- **{t}** Planning complete. Next phase: {e['next_phase']}."
    return f"- **{t}** {ev}: {json.dumps({k: v for k, v in e.items() if k not in ('ts', 'event')})}"


def changelog_md(log: list[dict]) -> str:
    return "# Planning changelog\n\nEvery decision in this run, oldest first.\n\n" + "\n".join(_line(e) for e in log) + "\n"
