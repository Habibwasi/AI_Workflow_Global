"""Checkpointers (run history) and git commits of approved specs.

Checkpointers, from the LangGraph docs:
    InMemorySaver   (langgraph-checkpoint)          - tests only, lost when the process stops
    SqliteSaver     (langgraph-checkpoint-sqlite)   - local work, one file
    PostgresSaver   (langgraph-checkpoint-postgres) - production

With a persistent checkpointer you can resume a run later (same thread id), list
every step with ``graph.get_state_history(config)``, replay from a past checkpoint,
or fork from one with ``graph.update_state(...)``.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
from contextlib import contextmanager
from typing import Optional


def sqlite_checkpointer(path: str = ".workflow/checkpoints.sqlite"):
    """File-backed checkpointer. The file survives restarts; reuse the thread id to resume."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


@contextmanager
def postgres_checkpointer(conn_string: str):
    """Production checkpointer. Needs: pip install langgraph-checkpoint-postgres."""
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(conn_string) as saver:
        saver.setup()
        yield saver


# --------------------------------------------------------------------------- #
# git: one commit per approval, reason in the message
# --------------------------------------------------------------------------- #
def _git(repo: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


def check_repo(repo: str) -> None:
    if _git(repo, "rev-parse", "--is-inside-work-tree").returncode != 0:
        raise ValueError(f"specs_repo is not a git repository: {repo}")


def commit_specs(repo: str, specs_dir: str, files: dict[str, str], subject: str, body: str) -> Optional[str]:
    """Write files under <repo>/<specs_dir>/ and commit them. Returns the sha, or None
    when nothing changed (makes a re-run of the same step harmless)."""
    base = os.path.join(repo, specs_dir)
    os.makedirs(base, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(base, name), "w", encoding="utf-8") as f:
            f.write(text)
    _git(repo, "add", "--", specs_dir)
    if _git(repo, "diff", "--cached", "--quiet", "--", specs_dir).returncode == 0:
        return None
    res = _git(repo, "commit", "-m", subject, "-m", body, "--", specs_dir)
    if res.returncode != 0:
        raise RuntimeError(f"git commit failed: {res.stderr.strip() or res.stdout.strip()}")
    return _git(repo, "rev-parse", "HEAD").stdout.strip()
