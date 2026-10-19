"""LangSmith observability.

Tracing itself needs no code: with LANGSMITH_TRACING=true and LANGSMITH_API_KEY set,
every node and model call is sent to LangSmith automatically. This module only
makes the traces easier to use:

* ``trace_config``   names the top-level run and adds tags/metadata to filter by
                     (part, mode, project, workflow version). LangGraph already adds
                     the run's ``thread_id``, which LangSmith uses to group every
                     resume of one planning run into a single thread.
* ``tracing_scope``  with ``hide_content=True`` sends timings, tokens, errors and
                     metadata but strips every input and output (prompts, your
                     answers, the SRD text). Use it for sensitive projects.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from . import __version__

PART = "planning"


def tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")


def trace_config(*, mode: str, project: str, part: str = PART) -> dict:
    """Extra keys to merge into the config passed to graph.invoke()."""
    return {
        "run_name": part,
        "tags": [f"part:{part}", f"mode:{mode}", f"project:{project}"],
        "metadata": {"part": part, "mode": mode, "project": project, "workflow_version": __version__},
    }


def _strip(_: dict) -> dict:
    return {}


@contextmanager
def tracing_scope(hide_content: bool = False) -> Iterator[None]:
    """Wrap every graph.invoke() of a run in this.

    Uses a LangSmith client whose hide_inputs/hide_outputs drop all content, so
    nothing but metadata leaves the machine. Does nothing when tracing is off.
    """
    if not (hide_content and tracing_enabled()):
        yield
        return
    from langsmith import Client, tracing_context

    client = Client(hide_inputs=_strip, hide_outputs=_strip)
    try:
        with tracing_context(client=client):
            yield
    finally:
        client.flush()
