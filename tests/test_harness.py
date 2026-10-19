"""Subscription back ends (Claude Code CLI, Copilot CLI) against stand-in CLIs.

No real CLI, account or network is needed: tests/fake_cli.py plays both tools and
records exactly how they were called.
"""
from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from planning import Settings, build_graph, initial_state
from planning.harness import (STEPS, ClaudeCodeLLM, CopilotCLILLM, HarnessError, RoutedLLM, build_llm,
                              extract_json, load_routing)
from planning.state import SRD, TRD, ClarifyResult, RejectionClassification, ScopeDoc, TestabilityResult

from .test_flow import APPROVE, FEAS, REQ1, REQ2, SCOPE, TC1, TC2, drive, reject

pytestmark = pytest.mark.skipif(os.name == "nt", reason="stand-in CLIs are POSIX shell wrappers")

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_cli.py")
CLASSIFICATION = {"kind": "detail", "affected_req_ids": ["REQ-002"], "rationale": "one criterion"}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Install stand-in CLIs and return a helper to script their replies."""
    base = tmp_path / "fake"
    for cli in ("claude", "copilot"):
        (base / cli).mkdir(parents=True)
        wrapper = tmp_path / f"bin_{cli}"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" {cli} "$@"\n')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv(f"{cli.upper()}_CLI", str(wrapper))
    monkeypatch.setenv("FAKE_DIR", str(base))

    class Fake:
        workdir = str(tmp_path / "harness")

        def reply(self, cli: str, schema: str, *replies: tuple[str, object]):
            (base / cli / f"{schema}.json").write_text(json.dumps([{"mode": m, "data": d} for m, d in replies]))

        def calls(self) -> list[dict]:
            path = base / "calls.jsonl"
            return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    return Fake()


def dump(model):
    return model.model_dump()


# --------------------------------------------------------------------------- #
# Claude Code
# --------------------------------------------------------------------------- #
def test_claude_call_shape_uses_plan_not_api_key(fake, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-be-used")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "plan-token")
    fake.reply("claude", "RejectionClassification", ("ok", CLASSIFICATION))

    out = ClaudeCodeLLM(model="sonnet", workdir=fake.workdir).structured(
        RejectionClassification, "SYSTEM RULES", "USER PROMPT TEXT")

    assert out == RejectionClassification(**CLASSIFICATION)
    (call,) = fake.calls()
    argv = call["argv"]
    assert argv[0] == "-p" and "--bare" not in argv               # bare mode would skip the plan sign-in
    assert argv[argv.index("--output-format") + 1] == "json"
    assert json.loads(argv[argv.index("--json-schema") + 1])["title"] == "RejectionClassification"
    assert argv[argv.index("--system-prompt") + 1] == "SYSTEM RULES"
    assert argv[argv.index("--tools") + 1] == ""                    # no tools at all
    assert argv[argv.index("--disallowedTools") + 1] == "mcp__*"
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert "--no-session-persistence" in argv and "--strict-mcp-config" in argv
    assert call["stdin"] == "USER PROMPT TEXT"                      # prompt via stdin, not the command line
    assert call["env"]["ANTHROPIC_API_KEY"] is None and call["env"]["ANTHROPIC_BASE_URL"] is None
    assert call["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "plan-token"   # plan token kept
    assert os.path.realpath(call["cwd"]) == os.path.realpath(fake.workdir)


def test_claude_falls_back_to_parsing_the_text_answer(fake):
    fake.reply("claude", "RejectionClassification", ("text", CLASSIFICATION))
    out = ClaudeCodeLLM(workdir=fake.workdir).structured(RejectionClassification, "s", "u")
    assert out.kind == "detail"


def test_claude_usage_limit_error_says_how_to_resume(fake):
    fake.reply("claude", "ScopeDoc", ("error", None))
    with pytest.raises(HarnessError, match="usage limit.*--thread"):
        ClaudeCodeLLM(workdir=fake.workdir).structured(ScopeDoc, "s", "u")


def test_claude_retries_once_with_the_validation_error(fake):
    fake.reply("claude", "RejectionClassification", ("garbage", None), ("ok", CLASSIFICATION))
    llm = ClaudeCodeLLM(workdir=fake.workdir)
    assert llm.structured(RejectionClassification, "s", "u").kind == "detail"
    second = fake.calls()[1]["stdin"]
    assert "YOUR PREVIOUS ANSWER WAS INVALID" in second and llm.calls == 2


# --------------------------------------------------------------------------- #
# Copilot
# --------------------------------------------------------------------------- #
def test_copilot_call_shape_and_fenced_json(fake):
    fake.reply("copilot", "RejectionClassification", ("text", CLASSIFICATION))
    out = CopilotCLILLM(model="gpt-5-mini", workdir=fake.workdir).structured(
        RejectionClassification, "SYSTEM RULES", "USER PROMPT TEXT")
    assert out.affected_req_ids == ["REQ-002"]

    (call,) = fake.calls()
    argv = call["argv"]
    prompt = argv[argv.index("-p") + 1]
    assert prompt.startswith("SYSTEM RULES\n\nUSER PROMPT TEXT") and "JSON SCHEMA:" in prompt
    assert {"-s", "--no-ask-user", "--deny-tool=shell", "--deny-tool=write"} <= set(argv)
    assert argv[argv.index("--model") + 1] == "gpt-5-mini"


def test_copilot_retries_once_then_gives_up(fake):
    fake.reply("copilot", "RejectionClassification", ("garbage", None))
    llm = CopilotCLILLM(workdir=fake.workdir)
    with pytest.raises(HarnessError, match="did not return a valid RejectionClassification after 2 attempts"):
        llm.structured(RejectionClassification, "s", "u")
    assert llm.calls == 2                     # each attempt uses Copilot credits, so retries are capped


def test_copilot_failure_message_is_surfaced(fake):
    fake.reply("copilot", "ScopeDoc", ("error", None))
    with pytest.raises(HarnessError, match="copilot failed.*usage limit"):
        CopilotCLILLM(workdir=fake.workdir).structured(ScopeDoc, "s", "u")


def test_copilot_refuses_prompts_too_long_for_windows(fake, monkeypatch):
    import planning.harness as h
    monkeypatch.setattr(h.os, "name", "nt")
    with pytest.raises(HarnessError, match="too long for a Windows command line"):
        CopilotCLILLM(workdir=fake.workdir).structured(ScopeDoc, "s", "x" * 40_000)
    assert fake.calls() == []


# --------------------------------------------------------------------------- #
# Routing and setup errors
# --------------------------------------------------------------------------- #
def test_missing_cli_explains_how_to_install(monkeypatch, tmp_path):
    monkeypatch.delenv("CLAUDE_CLI", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(HarnessError, match="claude' was not found on PATH.*npm install -g @anthropic-ai/claude-code"):
        ClaudeCodeLLM(workdir=str(tmp_path)).structured(ScopeDoc, "s", "u")


def test_routing_file(tmp_path):
    path = tmp_path / "routing.json"
    path.write_text(json.dumps({"default": "claude:sonnet", "steps": {"classify": "copilot:gpt-5-mini"}}))
    r = load_routing("api", str(path))
    assert isinstance(r.default, ClaudeCodeLLM) and r.default.model == "sonnet"
    assert isinstance(r.routes["RejectionClassification"], CopilotCLILLM)
    assert r.describe()["classify"] == "copilot:gpt-5-mini" and r.describe()["srd"] == "claude:sonnet"

    path.write_text(json.dumps({"steps": {"planning": "copilot"}}))
    with pytest.raises(ValueError, match="Unknown step"):
        load_routing("claude", str(path))
    with pytest.raises(ValueError, match="Unknown back end"):
        build_llm("gemini")


def test_example_routing_file_is_valid():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = load_routing("api", os.path.join(root, "routing.example.json"))
    assert set(r.describe()) == set(STEPS)


def test_extract_json_handles_prose_and_fences():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Result: {"a": {"b": 2}} done') == {"a": {"b": 2}}


# --------------------------------------------------------------------------- #
# Whole planning run on the two harnesses
# --------------------------------------------------------------------------- #
def test_full_planning_run_routed_across_claude_and_copilot(fake, tmp_path):
    fake.reply("claude", "ClarifyResult", ("ok", dump(ClarifyResult())))
    fake.reply("claude", "ScopeDoc", ("ok", dump(SCOPE)))
    fake.reply("claude", "FeasibilityReport", ("ok", dump(FEAS)))
    fake.reply("claude", "SRD", ("ok", dump(SRD(requirements=[REQ1, REQ2]))))
    fake.reply("claude", "TRD", ("ok", dump(TRD(test_cases=[TC1, TC2]))))
    fake.reply("copilot", "TestabilityResult", ("text", dump(TestabilityResult())))
    fake.reply("copilot", "RejectionClassification", ("ok", CLASSIFICATION))

    routing = tmp_path / "routing.json"
    routing.write_text(json.dumps({"default": "claude",
                                   "steps": {"classify": "copilot", "testability": "copilot"}}))
    llm = load_routing("api", str(routing), workdir=fake.workdir)
    g = build_graph(llm, Settings(output_dir=str(tmp_path / "out")))
    result, payloads = drive(g, initial_state("Build password reset"),
                             [APPROVE, reject("REQ-002 should expire after 30 minutes"), APPROVE, APPROVE])

    assert result["status"] == "handed_off"
    handled = {(c["cli"], c["schema"]) for c in fake.calls()}
    assert handled == {("claude", "ClarifyResult"), ("claude", "ScopeDoc"), ("claude", "FeasibilityReport"),
                       ("claude", "SRD"), ("claude", "TRD"),
                       ("copilot", "TestabilityResult"), ("copilot", "RejectionClassification")}
    cls = [e for e in result["audit_log"] if e["event"] == "rejection_classified"][0]
    assert cls["kind"] == "detail" and cls["affected_req_ids"] == ["REQ-002"]


def test_error_hints_for_limits_and_safety_filter_false_positives():
    from planning.harness import _limit_hint
    assert "--thread" in _limit_hint("Claude AI usage limit reached")
    assert "false positive" in _limit_hint("API Error: Sonnet's safeguards flagged this message")
    assert _limit_hint("network unreachable") == ""
