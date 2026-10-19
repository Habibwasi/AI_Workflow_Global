"""Interactive terminal driver for Part 1.

    # on your subscriptions (default): Claude Code and Copilot CLI, signed in once
    python run_cli.py --mode requirements_first "Build a password reset flow for our web app"
    cp routing.example.json routing.json     # choose claude or copilot per step

    # or on an API key
    python run_cli.py "..." --llm api:anthropic:claude-sonnet-5-5

    # commit every approved spec into your project's git repo, with the reasons
    python run_cli.py "..." --specs-repo ../my-project

    # stopped half-way? resume the same run later (state lives in .workflow/checkpoints.sqlite)
    python run_cli.py --thread <run id>

    # list every saved step of a run
    python run_cli.py --thread <run id> --history

    # LangSmith observability (no code changes; see README)
    export LANGSMITH_TRACING=true LANGSMITH_API_KEY=... LANGSMITH_PROJECT=ai-workflow
    python run_cli.py "..." --no-content     # sensitive project: metadata only

Every pause (question, review, escalation) is printed; you answer in the terminal and
the graph resumes from its checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid

from langgraph.types import Command

from planning import ChatModelLLM, Settings, build_graph, initial_state
from planning.harness import HarnessError, load_routing
from planning.persistence import sqlite_checkpointer
from planning.tracing import trace_config, tracing_enabled, tracing_scope


def ask_user(payload: dict) -> dict:
    kind = payload.get("type")
    if payload.get("error"):
        print(f"\n! {payload['error']}")

    if kind == "clarify":
        print("\n=== Questions from the planning agent ===")
        answers = {}
        for q in payload["questions"]:
            print(f"\n[{q['id']}] {q['question']}\n    (why: {q['why_it_matters']})")
            answers[q["id"]] = input("> ").strip()
        return answers

    if kind == "review":
        print(f"\n=== Review: {payload['gate'].upper()} ===\n")
        print(payload["artifact"])
        for key in ("feasibility",):
            if payload.get(key):
                print(payload[key])
        for key in ("addressing_feedback", "open_testability_issues", "unresolved_questions",
                    "requirements_without_tests"):
            if payload.get(key):
                print(f"\n{key}: {json.dumps(payload[key], indent=2)}")
        choice = input("\n[a]pprove or [r]eject? ").strip().lower()
        if choice.startswith("a"):
            return {"action": "approve"}
        return {"action": "reject", "feedback": input("What is wrong? ").strip()}

    if kind == "escalation":
        print(f"\n=== Escalation: gate '{payload['gate']}' rejected {payload['rejections_at_gate']} times ===")
        for fb in payload["recent_feedback"]:
            print(f"- {fb}")
        choice = input("\n[c]ontinue or [x] abort? ").strip().lower()
        if choice.startswith("x"):
            return {"action": "abort"}
        return {"action": "continue", "guidance": input("Extra guidance (optional): ").strip()}

    print(json.dumps(payload, indent=2))
    return json.loads(input("Resume JSON> "))


def pending_interrupts(graph, config) -> list:
    """Interrupts a stopped run is waiting on (empty if the thread is new or finished)."""
    snap = graph.get_state(config)
    return [i for task in snap.tasks for i in task.interrupts] if snap.next else []


def print_history(graph, config) -> None:
    snaps = list(graph.get_state_history(config))[::-1]          # oldest first
    if not snaps:
        print("No history for this thread.")
        return
    for s in snaps:
        step = (s.metadata or {}).get("step")
        nxt = ", ".join(s.next) or "(end)"
        print(f"{s.created_at}  step {step:>3}  next: {nxt}  checkpoint: {s.config['configurable']['checkpoint_id']}")


def build_planning_llm(args):
    """--llm/--routing pick the back end per step; --model alone keeps the old API behaviour."""
    if args.model and not (args.llm or args.routing):
        return ChatModelLLM(args.model), {"*": f"api:{args.model}"}
    routing = args.routing or ("routing.json" if os.path.exists("routing.json") else None)
    llm = load_routing(args.llm or "claude", routing)
    return llm, llm.describe()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", help="What to build (omit when resuming with --thread)")
    p.add_argument("--mode", choices=["requirements_first", "design_first"], default="requirements_first")
    p.add_argument("--llm", default=None,
                   help="Default back end for every step: claude[:model] (your Claude plan), "
                        "copilot[:model] (your Copilot plan) or api[:provider:model]. Default: claude")
    p.add_argument("--routing", default=None,
                   help="JSON file choosing a back end per step (default: ./routing.json if present)")
    p.add_argument("--model", default=None, help="API model (provider:model). Same as --llm api:<model>")
    p.add_argument("--thread", default=None, help="Thread id of a run to resume or inspect")
    p.add_argument("--db", default=".workflow/checkpoints.sqlite", help="SQLite file holding run history")
    p.add_argument("--specs-repo", default=None, help="Project git repo: commit approved specs there")
    p.add_argument("--history", action="store_true", help="List every saved step of --thread and exit")
    p.add_argument("--max-rejections", type=int, default=3)
    p.add_argument("--project", default=None,
                   help="Project name for LangSmith filtering (default: folder name of --specs-repo or the current folder)")
    p.add_argument("--no-content", action="store_true",
                   help="LangSmith gets timings, tokens, errors and tags only: no prompts, answers or spec text")
    args = p.parse_args()

    settings = Settings(max_rejections=args.max_rejections, specs_repo=args.specs_repo)
    llm, routes = (None, {}) if args.history else build_planning_llm(args)
    graph = build_graph(llm, settings, checkpointer=sqlite_checkpointer(args.db))
    thread = args.thread or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 200}

    if args.history:
        print_history(graph, config)
        return

    snap = graph.get_state(config)
    pending = pending_interrupts(graph, config)
    failed = bool(snap.next) and not pending          # a step raised (e.g. a plan's usage limit): retry it
    if not (pending or failed) and not args.prompt:
        p.error("give a prompt to start a run, or --thread of a run that is waiting for you")
    saved = snap.values if (pending or failed) else {}
    mode = saved.get("mode", args.mode)
    hide = args.no_content or bool(saved.get("hide_trace_content"))   # sticky for the whole run
    project = args.project or os.path.basename(os.path.abspath(args.specs_repo or os.getcwd()))
    config.update(trace_config(mode=mode, project=project))
    if tracing_enabled():
        what = "metadata only (--no-content)" if hide else "full content"
        print(f"LangSmith tracing on: project '{os.environ.get('LANGSMITH_PROJECT', 'default')}', {what}")

    if routes:
        print("Model per step: " + ", ".join(f"{step}={spec}" for step, spec in routes.items()))

    try:
        with tracing_scope(hide_content=hide):
            if pending:
                print(f"Resuming run {thread}")
                result = {"__interrupt__": pending}
            elif failed:
                print(f"Resuming run {thread}: retrying {', '.join(snap.next)}")
                result = graph.invoke(None, config)
            else:
                print(f"Run id: {thread}  (resume later with --thread {thread})")
                result = graph.invoke(initial_state(args.prompt, mode, hide_trace_content=hide), config)

            while "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                result = graph.invoke(Command(resume=ask_user(payload)), config)
    except HarnessError as e:
        print(f"\nStopped: {e}\nEverything up to this step is saved. Resume with: "
              f"python run_cli.py --thread {thread}")
        raise SystemExit(1)

    print(f"\nStatus: {result['status']}")
    if result.get("artifact_paths"):
        print(f"Next phase: {result['next_phase']}")
        for name, path in result["artifact_paths"].items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
