"""Interactive terminal driver for Part 1.

    export ANTHROPIC_API_KEY=...            # or another provider's key
    export PLANNING_MODEL=anthropic:claude-sonnet-5-5
    python run_cli.py --mode requirements_first "Build a password reset flow for our web app"

Every pause (question, review, escalation) is printed; you answer in the terminal and
the graph resumes from its checkpoint.
"""
from __future__ import annotations

import argparse
import json
import uuid

from langgraph.types import Command

from planning import ChatModelLLM, Settings, build_graph, initial_state


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--mode", choices=["requirements_first", "design_first"], default="requirements_first")
    p.add_argument("--model", default=None)
    p.add_argument("--thread", default=None, help="Reuse a thread id to resume (needs a persistent checkpointer)")
    p.add_argument("--max-rejections", type=int, default=3)
    args = p.parse_args()

    graph = build_graph(ChatModelLLM(args.model), Settings(max_rejections=args.max_rejections))
    config = {"configurable": {"thread_id": args.thread or str(uuid.uuid4())}, "recursion_limit": 200}

    result = graph.invoke(initial_state(args.prompt, args.mode), config)
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
