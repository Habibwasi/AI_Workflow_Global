"""Subscription back ends: run the planning steps on your Claude and Copilot plans.

Instead of calling a model API, each planning step hands its prompt to a coding
harness's command-line tool, which is signed in with your subscription, and reads
the answer back into the LangGraph state. The workflow itself does not change:
every node still calls ``llm.structured(schema, system, user)``.

    ClaudeCodeLLM   ``claude -p``   (Claude Pro/Max plan, via Claude Code)
    CopilotCLILLM   ``copilot -p``  (GitHub Copilot plan, via Copilot CLI)
    RoutedLLM       picks a back end per planning step (see ``load_routing``)

Why the command-line tools and not the chat panels in Visual Studio or VS Code:
no IDE chat panel can be driven by another program. The CLIs use the same
subscriptions and the same sign-in.

Safety: planning calls run in a neutral, empty folder (``~/.ai_workflow/harness``)
so they never pick up a project's CLAUDE.md, hooks or MCP servers, and they get no
tools: they can't run commands, edit files or browse. API-key variables are removed
from the Claude call so it always uses your plan and never bills a key by accident.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_WORKDIR = Path.home() / ".ai_workflow" / "harness"

# Variables that would make Claude Code bill an API key or another provider
# instead of your Claude plan. CLAUDE_CODE_OAUTH_TOKEN (a plan token from
# `claude setup-token`) is deliberately kept.
_NON_PLAN_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                 "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY")

# Friendly step names (used in routing.json) -> the schema each step produces.
STEPS = {
    "clarify": "ClarifyResult",
    "scope": "ScopeDoc",
    "feasibility": "FeasibilityReport",
    "srd": "SRD",
    "testability": "TestabilityResult",
    "classify": "RejectionClassification",
    "trd": "TRD",
    "design": "DesignDoc",
    "consistency": "ConsistencyResult",
}

WINDOWS_ARG_LIMIT = 30_000   # CreateProcess caps a command line at 32,767 characters


class HarnessError(RuntimeError):
    """A harness call failed. The message says what to do next."""


def _exe(name: str, env_override: str) -> str:
    path = os.environ.get(env_override) or shutil.which(name)
    if not path:
        hint = {"claude": "npm install -g @anthropic-ai/claude-code, then run `claude` once and sign in with your plan",
                "copilot": "npm install -g @github/copilot, then run `copilot` once and sign in"}[name]
        raise HarnessError(f"'{name}' was not found on PATH. Install it: {hint}.")
    return path


def _workdir(path: str | os.PathLike | None) -> str:
    p = Path(path) if path else DEFAULT_WORKDIR
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def extract_json(text: str):
    """Find the JSON object in a model's text answer (tolerates code fences and prose)."""
    s = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end > start:
            return json.loads(s[start:end + 1])
        raise


def _traceable(name: str) -> Callable:
    """Show each harness call as its own step in LangSmith (no-op when tracing is off)."""
    try:
        from langsmith import traceable
    except ImportError:                                   # pragma: no cover
        return lambda f: f
    return traceable(name=name, run_type="llm")


def _limit_hint(message: str) -> str:
    if re.search(r"usage limit|rate limit|quota|credits", message, re.I):
        return (" Your plan's usage limit looks exhausted. The run is saved: resume it later "
                "with --thread <run id>, or route this step to the other harness.")
    if re.search(r"safeguards flagged|usage policy|aup", message, re.I):
        return (" A safety filter flagged this request; on ordinary planning text that is usually a "
                "false positive. Resume to retry, or route this step to another model in routing.json "
                "(e.g. claude:opus or copilot).")
    return ""


class _HarnessLLM:
    """Shared retry/validation loop. Subclasses implement ``_ask``."""

    name = "harness"

    def __init__(self, model: str | None = None, workdir: str | os.PathLike | None = None,
                 timeout: int = 600, max_attempts: int = 2):
        self.model = model
        self.workdir = workdir
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.calls = 0          # how many CLI calls this back end has made (each one uses plan quota)

    def structured(self, schema: type[T], system: str, user: str) -> T:
        schema_json = json.dumps(schema.model_json_schema())
        prompt, last_error = user, None
        for attempt in range(1, self.max_attempts + 1):
            raw = self._traced_ask(schema.__name__, schema_json, system, prompt)
            try:
                return schema.model_validate(raw if isinstance(raw, dict) else extract_json(raw))
            except (ValidationError, json.JSONDecodeError, TypeError) as e:
                last_error = str(e)[:1500]
                prompt = (f"{user}\n\nYOUR PREVIOUS ANSWER WAS INVALID:\n{last_error}\n"
                          f"Answer again with one JSON object that matches the schema exactly.")
        raise HarnessError(f"{self.name} did not return a valid {schema.__name__} after "
                           f"{self.max_attempts} attempts. Last error: {last_error}")

    def _traced_ask(self, schema_name: str, schema_json: str, system: str, prompt: str):
        return _traceable(f"{self.name}:{schema_name}")(self._ask)(schema_json, system, prompt)

    def _run(self, cmd: list[str], stdin: str | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
        self.calls += 1
        try:
            return subprocess.run(cmd, input=stdin, cwd=_workdir(self.workdir), env=env, capture_output=True,
                                  text=True, encoding="utf-8", timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            raise HarnessError(f"{self.name} did not answer within {self.timeout}s.") from e

    def _ask(self, schema_json: str, system: str, prompt: str):   # pragma: no cover
        raise NotImplementedError


class ClaudeCodeLLM(_HarnessLLM):
    """Planning calls through ``claude -p`` on your Claude plan.

    Uses Claude Code's own schema enforcement (``--json-schema``), so answers come
    back already shaped. Not ``--bare``: bare mode skips your plan sign-in and
    needs an API key.
    """

    name = "claude-code"

    def _ask(self, schema_json: str, system: str, prompt: str):
        cmd = [_exe("claude", "CLAUDE_CLI"), "-p",
               "--output-format", "json",
               "--json-schema", schema_json,
               "--system-prompt", system,
               "--tools", "",                       # no built-in tools at all
               "--disallowedTools", "mcp__*",       # and no MCP tools
               "--strict-mcp-config",
               "--disable-slash-commands",
               "--permission-mode", "dontAsk",
               "--no-session-persistence"]
        if self.model:
            cmd += ["--model", self.model]
        env = {k: v for k, v in os.environ.items() if k not in _NON_PLAN_ENV}
        p = self._run(cmd, stdin=prompt, env=env)   # prompt on stdin: no command-line length limit

        try:
            data = json.loads(p.stdout)
        except json.JSONDecodeError:
            detail = (p.stderr or p.stdout or "").strip()[-1500:]
            raise HarnessError(f"claude failed (exit {p.returncode}): {detail}{_limit_hint(detail)}")
        if p.returncode != 0 or data.get("is_error"):
            msg = str(data.get("result") or p.stderr or "unknown error")[:1500]
            raise HarnessError(f"claude reported an error: {msg}{_limit_hint(msg)}")

        self.last_usage = data.get("usage")
        if data.get("structured_output") is not None:
            return data["structured_output"]
        return data.get("result", "")             # fall back to parsing the text answer


class CopilotCLILLM(_HarnessLLM):
    """Planning calls through ``copilot -p`` on your GitHub Copilot plan.

    Copilot CLI has no schema option, so the schema goes into the prompt and the
    answer is validated here; an invalid answer is retried once with the error.
    Each call, retries included, uses Copilot AI credits.
    """

    name = "copilot"

    def _ask(self, schema_json: str, system: str, prompt: str):
        full = (f"{system}\n\n{prompt}\n\n"
                "Respond with only one JSON object that matches this JSON Schema. "
                "No prose before or after it, no code fences.\n"
                f"JSON SCHEMA:\n{schema_json}")
        if os.name == "nt" and len(full) > WINDOWS_ARG_LIMIT:
            raise HarnessError(f"This prompt is {len(full):,} characters, too long for a Windows command line. "
                               "Route this step to claude in routing.json (it reads prompts from stdin).")
        cmd = [_exe("copilot", "COPILOT_CLI"), "-p", full, "-s", "--no-ask-user",
               "--deny-tool=shell", "--deny-tool=write"]
        if self.model:
            cmd += ["--model", self.model]
        p = self._run(cmd)
        if p.returncode != 0:
            detail = (p.stderr or p.stdout or "").strip()[-1500:]
            raise HarnessError(f"copilot failed (exit {p.returncode}): {detail}{_limit_hint(detail)}")
        return p.stdout


class RoutedLLM:
    """Sends each planning step to the back end chosen for it."""

    def __init__(self, default, routes: dict | None = None, labels: dict | None = None):
        self.default = default
        self.routes = routes or {}          # schema name -> back end
        self.labels = labels or {}          # schema name -> spec string, for display

    def structured(self, schema: type[T], system: str, user: str) -> T:
        return self.routes.get(schema.__name__, self.default).structured(schema, system, user)

    def describe(self) -> dict[str, str]:
        return {step: self.labels.get(s, self.labels.get("*", "?")) for step, s in STEPS.items()}


def build_llm(spec: str, workdir: str | None = None):
    """``claude``, ``claude:<model>``, ``copilot``, ``copilot:<model>``, ``api`` or ``api:<provider:model>``."""
    kind, _, model = spec.strip().partition(":")
    model = model or None
    if kind == "claude":
        return ClaudeCodeLLM(model=model, workdir=workdir)
    if kind == "copilot":
        return CopilotCLILLM(model=model, workdir=workdir)
    if kind == "api":
        from .llm import ChatModelLLM
        return ChatModelLLM(model)
    raise ValueError(f"Unknown back end '{spec}'. Use claude[:model], copilot[:model] or api[:provider:model].")


def load_routing(default_spec: str, path: str | None = None, workdir: str | None = None) -> RoutedLLM:
    """Build a RoutedLLM from a default spec plus an optional routing.json:

        {"default": "claude", "steps": {"classify": "copilot", "testability": "copilot"}}

    The file's "default" wins over ``default_spec`` when present.
    """
    steps: dict[str, str] = {}
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        default_spec = cfg.get("default", default_spec)
        steps = cfg.get("steps", {})
        unknown = sorted(set(steps) - set(STEPS))
        if unknown:
            raise ValueError(f"Unknown step(s) in {path}: {', '.join(unknown)}. Valid: {', '.join(STEPS)}")

    cache: dict[str, object] = {}

    def get(spec: str):
        if spec not in cache:
            cache[spec] = build_llm(spec, workdir)
        return cache[spec]

    routes = {STEPS[step]: get(spec) for step, spec in steps.items()}
    labels = {"*": default_spec, **{STEPS[step]: spec for step, spec in steps.items()}}
    return RoutedLLM(get(default_spec), routes, labels)
