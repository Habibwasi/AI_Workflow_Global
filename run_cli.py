"""Interactive terminal driver for Part 1.

    export ANTHROPIC_API_KEY=...            # or another provider's key
    export PLANNING_MODEL=anthropic:claude-sonnet-5-5
    python run_cli.py --mode requirements_first "Build a password reset flow for our web app"

    # commit every approved spec into your project's git repo, with the reasons
    python run_cli.py "..." --specs-repo ../my-project

    # stopped half-way? resume the same run later (state lives in .workflow/checkpoints.sqlite)
    python run_cli.py --thread <run id>

    # list every saved step of a run
    python run_cli.py --thread <run id> --history

Every pause (question, review, escalation) is printed; you answer in the terminal and
the graph resumes from its checkpoint.
"""
from __future__ import annotations

import argparse
import json
import uuid

from langgraph.types import Command

from planning import ChatModelLLM, Settings, build_graph, initial_state
from planning.persistence import sqlite_checkpointer


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", help="What to build (omit when resuming with --thread)")
    p.add_argument("--mode", choices=["requirements_first", "design_first"], default="requirements_first")
    p.add_argument("--model", default=None)
    p.add_argument("--thread", default=None, help="Thread id of a run to resume or inspect")
    p.add_argument("--db", default=".workflow/checkpoints.sqlite", help="SQLite file holding run history")
    p.add_argument("--specs-repo", default=None, help="Project git repo: commit approved specs there")
    p.add_argument("--history", action="store_true", help="List every saved step of --thread and exit")
    p.add_argument("--max-rejections", type=int, default=3)
    args = p.parse_args()

    settings = Settings(max_rejections=args.max_rejections, specs_repo=args.specs_repo)
    graph = build_graph(ChatModelLLM(args.model) if not args.history else None, settings,
                        checkpointer=sqlite_checkpointer(args.db))
    thread = args.thread or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 200}

    if args.history:
        print_history(graph, config)
        return

    pending = pending_interrupts(graph, config)
    if pending:
        print(f"Resuming run {thread}")
        result = {"__interrupt__": pending}
    elif args.prompt:
        print(f"Run id: {thread}  (resume later with --thread {thread})")
        result = graph.invoke(initial_state(args.prompt, args.mode), config)
    else:
        p.error("give a prompt to start a run, or --thread of a run that is waiting for you")

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        result = graph.invoke(Command(resume=ask_user(payload)), config)

    print(f"\nStatus: {result['status']}")
    if result.get("artifact_paths"):
        print(f"Next phase: {result['next_phase']}")
        for name, path in result["artifact_paths"].items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
