"""Audit trail, git commits per approval, and run history that survives a restart."""
from __future__ import annotations

import json
import os
import subprocess
import uuid

import pytest
from langgraph.types import Command

from planning import Settings, build_graph, initial_state
from planning.persistence import check_repo, commit_specs, sqlite_checkpointer
from planning.state import SRD, TRD, RejectionClassification

from .test_flow import (APPROVE, REQ1, REQ1_TAMPERED, REQ2, REQ2_V2, REQ2_VAGUE, TC1, TC2_V2,
                        FakeLLM, base_script, drive, reject)

FEEDBACK = "REQ-002 should expire after 30 minutes"


def detail_rejection_llm():
    return FakeLLM(base_script(
        SRD=[SRD(requirements=[REQ1, REQ2]), SRD(requirements=[REQ1_TAMPERED, REQ2_V2])],
        RejectionClassification=[RejectionClassification(kind="detail", affected_req_ids=["REQ-002"],
                                                         rationale="Only the expiry criterion changes")],
        TRD=[TRD(test_cases=[TC1, TC2_V2])],
    ))


def events(result, name):
    return [e for e in result["audit_log"] if e["event"] == name]


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "project"
    repo.mkdir()
    for cmd in (["init", "-q", "-b", "main"], ["config", "user.name", "Test"], ["config", "user.email", "t@example.com"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True)
    return str(repo)


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


# --------------------------------------------------------------------------- #
def test_decision_log_explains_why_each_version_exists(tmp_path):
    g = build_graph(detail_rejection_llm(), Settings(output_dir=str(tmp_path)))
    result, _ = drive(g, initial_state("Build password reset"), [APPROVE, reject(FEEDBACK), APPROVE, APPROVE])

    seq = [e["event"] for e in result["audit_log"]]
    assert seq[0] == "run_started" and seq[-1] == "handoff"

    reviews = [(e["artifact"], e["version"], e["decision"], e["feedback"]) for e in events(result, "review")]
    assert reviews == [("Scope", 1, "approve", None), ("SRD", 1, "reject", FEEDBACK),
                       ("SRD", 2, "approve", None), ("TRD", 1, "approve", None)]

    cls = events(result, "rejection_classified")[0]
    assert (cls["kind"], cls["affected_req_ids"], cls["method"], cls["route"]) == \
           ("detail", ["REQ-002"], "LLM classifier", "draft_srd")
    assert cls["rationale"] == "Only the expiry criterion changes"

    srd_v2 = result["artifact_versions"]["srd"][1]
    assert FEEDBACK in srd_v2["reason"] and srd_v2["changed"] == ["REQ-002"]
    assert "30 minutes" in srd_v2["diff"] and "60 minutes" in srd_v2["diff"]

    out = os.path.dirname(result["artifact_paths"]["SRD.md"])
    changelog = open(os.path.join(out, "CHANGELOG.md")).read()
    assert FEEDBACK in changelog and "classified as **detail**" in changelog
    assert os.path.exists(os.path.join(out, "versions", "SRD_v1.md"))
    assert "30 minutes" in open(os.path.join(out, "versions", "SRD_v2.diff")).read()
    lines = open(os.path.join(out, "decisions.jsonl")).read().splitlines()
    assert len(lines) == len(result["audit_log"]) and all(json.loads(l)["ts"] for l in lines)


def test_answers_and_self_checks_are_logged_with_their_source(tmp_path):
    llm = FakeLLM(base_script(SRD=[SRD(requirements=[REQ1, REQ2_VAGUE]), SRD(requirements=[REQ1, REQ2])]))
    g = build_graph(llm, Settings(output_dir=str(tmp_path)))
    result, _ = drive(g, initial_state("Build password reset"),
                      [APPROVE, {"T-REQ-002": "within 2 seconds"}, APPROVE, APPROVE])

    check = events(result, "self_check")[0]
    assert check["check"] == "testability" and check["issues"][0]["req_id"] == "REQ-002"
    answer = events(result, "clarified")[0]["answers"][0]
    assert answer["answer"] == "within 2 seconds" and answer["source"] == "requirement check (REQ-002)"
    assert result["artifact_versions"]["srd"][1]["reason"].startswith("testability self-check")


def test_each_approval_is_a_git_commit_with_its_reason(tmp_path, git_repo):
    g = build_graph(detail_rejection_llm(), Settings(output_dir=str(tmp_path), specs_repo=git_repo))
    result, _ = drive(g, initial_state("Build password reset"), [APPROVE, reject(FEEDBACK), APPROVE, APPROVE])

    subjects = git(git_repo, "log", "--format=%s").splitlines()
    assert subjects == ["specs: planning complete", "specs: TRD v1 approved",
                        "specs: SRD v2 approved", "specs: Scope v1 approved"]

    srd_commit = git(git_repo, "log", "--format=%B", "-1", "--grep=SRD v2 approved")
    assert FEEDBACK in srd_commit and "Changed: REQ-002" in srd_commit and "Rejections at this gate: 1" in srd_commit

    for f in ("SRD.md", "srd.json", "TRD.md", "scope.md", "feasibility.md", "CHANGELOG.md", "decisions.jsonl"):
        assert os.path.exists(os.path.join(git_repo, "specs", f)), f
    assert "30 minutes" in open(os.path.join(git_repo, "specs", "SRD.md")).read()
    assert len(events(result, "committed")) == 4


def test_commit_is_skipped_when_nothing_changed(git_repo):
    files = {"SRD.md": "# SRD\n"}
    assert commit_specs(git_repo, "specs", files, "first", "body")
    assert commit_specs(git_repo, "specs", files, "again", "body") is None


def test_specs_repo_must_be_a_git_repo(tmp_path):
    with pytest.raises(ValueError):
        check_repo(str(tmp_path))
    with pytest.raises(ValueError):
        build_graph(FakeLLM({}), Settings(specs_repo=str(tmp_path)))


def test_run_resumes_after_restart_and_keeps_full_history(tmp_path):
    db = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 200}

    # first process: run until the scope review, then "stop"
    g1 = build_graph(FakeLLM(base_script()), Settings(output_dir=str(tmp_path)), checkpointer=sqlite_checkpointer(db))
    first = g1.invoke(initial_state("Build password reset"), config)
    assert first["__interrupt__"][0].value["gate"] == "scope"
    del g1

    # second process: new graph, new connection, same file and thread id
    g2 = build_graph(FakeLLM(base_script()), Settings(output_dir=str(tmp_path)), checkpointer=sqlite_checkpointer(db))
    result = g2.invoke(Command(resume=APPROVE), config)
    for answer in (APPROVE, APPROVE):
        result = g2.invoke(Command(resume=answer), config)
    assert result["status"] == "handed_off"

    history = list(g2.get_state_history(config))
    assert len(history) > 10
    assert any(snap.next == ("srd_review",) for snap in history)        # every step is inspectable
    assert history[-1].values == {} or history[-1].values.get("srd") is None
