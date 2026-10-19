"""Run a planning flow with LangSmith tracing pointed at a local capture server and
report what would have been sent. Used by test_tracing.py in a subprocess so
LangSmith's process-wide settings never leak into other tests.

    python -m tests.trace_probe show|hide   -> prints one JSON line
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

captured: list[bytes] = []


class _Capture(BaseHTTPRequestHandler):
    def _handle(self):
        n = int(self.headers.get("content-length") or 0)
        captured.append(self.rfile.read(n) if n else b"")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    do_GET = do_POST = do_PATCH = _handle

    def log_message(self, *args):
        pass


def _model_class():
    from langchain_core.language_models.chat_models import SimpleChatModel

    class ComputedAnswerModel(SimpleChatModel):
        """Builds its answer at call time, like a real model. (A canned-response fake
        would store the answer in its own settings, which traces record as model
        parameters, making the content check meaningless.)"""

        @property
        def _llm_type(self) -> str:
            return "computed-answer"

        def _call(self, messages, stop=None, run_manager=None, **kwargs) -> str:
            return "-".join(["SECRET", "MODEL", "ANSWER"])

    return ComputedAnswerModel


def main(mode: str) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Capture)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ.update(LANGSMITH_TRACING="true", LANGSMITH_API_KEY="test-key",
                      LANGSMITH_ENDPOINT=f"http://127.0.0.1:{server.server_port}",
                      LANGSMITH_PROJECT="probe")

    from langchain_core.tracers.langchain import wait_for_all_tracers
    from langgraph.types import Command

    from planning import build_graph, initial_state
    from planning.tracing import trace_config, tracing_scope
    from tests.test_flow import APPROVE, FakeLLM, base_script

    _ComputedAnswerModel = _model_class()
    graph = build_graph(FakeLLM(base_script()))
    config = {"configurable": {"thread_id": "THREAD-PROBE"}, "recursion_limit": 200,
              **trace_config(mode="requirements_first", project="demo")}
    with tracing_scope(hide_content=(mode == "hide")):
        result = graph.invoke(initial_state("SECRET-REQUEST build password reset"), config)
        while "__interrupt__" in result:
            result = graph.invoke(Command(resume=APPROVE), config)
        # a real LangChain model call, as ChatModelLLM makes in production
        _ComputedAnswerModel().invoke("SECRET-MODEL-PROMPT")
    wait_for_all_tracers()

    sent = b"".join(captured)
    needles = ["SECRET-REQUEST", "SECRET-MODEL-PROMPT", "SECRET-MODEL-ANSWER", "THREAD-PROBE",
               "part:planning", "mode:requirements_first", "project:demo", "workflow_version", "draft_srd",
               "test-key"]
    print(json.dumps({"status": result["status"], "bytes": len(sent),
                      "found": {n: n.encode() in sent for n in needles}}))


if __name__ == "__main__":
    main(sys.argv[1])
