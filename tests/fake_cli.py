"""Stand-in for the `claude` and `copilot` CLIs, used by test_harness.py.

    python fake_cli.py claude|copilot <the real CLI's arguments...>

Logs each call (argv, stdin, cwd, selected env vars) to $FAKE_DIR/calls.jsonl and
answers from $FAKE_DIR/<cli>/<SchemaTitle>.json, a list of replies used in order
(the last one repeats). Each reply is {"mode": ..., "data": {...}}:

    ok      claude: structured_output set      copilot: bare JSON on stdout
    text    claude: JSON only in result text   copilot: JSON inside a code fence with prose
    garbage answer that is not valid JSON for the schema
    error   claude: is_error with a usage-limit message   copilot: exit 1 with that message
"""
from __future__ import annotations

import json
import os
import sys


def schema_title(cli: str, argv: list[str]) -> str:
    if cli == "claude":
        return json.loads(argv[argv.index("--json-schema") + 1])["title"]
    prompt = argv[argv.index("-p") + 1]
    return json.loads(prompt.split("JSON SCHEMA:\n", 1)[1])["title"]


def next_reply(base: str, cli: str, title: str) -> dict:
    replies = json.load(open(os.path.join(base, cli, f"{title}.json")))
    counter = os.path.join(base, cli, f"{title}.count")
    n = int(open(counter).read()) if os.path.exists(counter) else 0
    open(counter, "w").write(str(n + 1))
    return replies[min(n, len(replies) - 1)]


def main() -> int:
    cli, argv = sys.argv[1], sys.argv[2:]
    base = os.environ["FAKE_DIR"]
    stdin = sys.stdin.read() if cli == "claude" else ""
    title = schema_title(cli, argv)
    with open(os.path.join(base, "calls.jsonl"), "a") as f:
        f.write(json.dumps({"cli": cli, "argv": argv, "stdin": stdin, "cwd": os.getcwd(), "schema": title,
                            "env": {k: os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
                                                                   "CLAUDE_CODE_OAUTH_TOKEN")}}) + "\n")
    reply = next_reply(base, cli, title)
    mode, data = reply["mode"], reply.get("data")
    limit_msg = "Claude AI usage limit reached. Your limit will reset at 5pm."

    if cli == "claude":
        out = {"type": "result", "subtype": "success", "is_error": False, "session_id": "s1",
               "usage": {"input_tokens": 10, "output_tokens": 5}, "total_cost_usd": 0.001}
        if mode == "ok":
            out.update(structured_output=data, result=json.dumps(data))
        elif mode == "text":
            out["result"] = f"Here you go:\n```json\n{json.dumps(data)}\n```"
        elif mode == "garbage":
            out["result"] = "I could not decide."
        elif mode == "error":
            out.update(is_error=True, subtype="error_during_execution", result=limit_msg)
        print(json.dumps(out))
        return 1 if mode == "error" else 0

    if mode == "error":
        print(limit_msg, file=sys.stderr)
        return 1
    if mode == "ok":
        print(json.dumps(data))
    elif mode == "text":
        print(f"Sure, here is the result:\n```json\n{json.dumps(data)}\n```\nLet me know!")
    elif mode == "garbage":
        print('{"kind": "maybe"}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
