"""LangSmith tracing: what is sent, with and without --no-content.

Each case runs in its own process against a local capture server (tests/trace_probe.py),
so no network access or LangSmith account is needed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from planning import __version__
from planning.tracing import trace_config, tracing_enabled, tracing_scope

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def probe(mode: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LANGSMITH_", "LANGCHAIN_"))}
    out = subprocess.run([sys.executable, "-m", "tests.trace_probe", mode], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


METADATA = ["THREAD-PROBE", "part:planning", "mode:requirements_first", "project:demo",
            "workflow_version", "draft_srd"]
CONTENT = ["SECRET-REQUEST", "SECRET-MODEL-PROMPT", "SECRET-MODEL-ANSWER"]


def test_full_traces_include_content_and_metadata():
    r = probe("show")
    assert r["status"] == "handed_off" and r["bytes"] > 0
    assert all(r["found"][k] for k in METADATA + CONTENT), r["found"]
    assert not r["found"]["test-key"]          # the API key is never part of the trace data


def test_no_content_sends_metadata_but_no_prompts_answers_or_specs():
    r = probe("hide")
    assert r["status"] == "handed_off" and r["bytes"] > 0
    assert all(r["found"][k] for k in METADATA), r["found"]
    assert not any(r["found"][k] for k in CONTENT), r["found"]


def test_trace_config_shape():
    c = trace_config(mode="design_first", project="shop")
    assert c["run_name"] == "planning"
    assert c["tags"] == ["part:planning", "mode:design_first", "project:shop"]
    assert c["metadata"] == {"part": "planning", "mode": "design_first", "project": "shop",
                             "workflow_version": __version__}


def test_scope_is_a_no_op_when_tracing_is_off(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert not tracing_enabled()
    with tracing_scope(hide_content=True):   # must not create a LangSmith client
        pass
