"""LLM adapter.

Every node talks to the model through ``PlanningLLM.structured`` so the graph is
model-agnostic and tests can swap in a scripted fake.
"""
from __future__ import annotations

import os
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.environ.get("PLANNING_MODEL", "anthropic:claude-sonnet-5-5")


class PlanningLLM(Protocol):
    def structured(self, schema: type[T], system: str, user: str) -> T: ...


class ChatModelLLM:
    """Wraps any LangChain chat model that supports structured output.

    ``model`` uses LangChain's ``provider:model`` string, e.g.
    ``anthropic:claude-sonnet-5-5`` or ``openai:gpt-...``.
    """

    def __init__(self, model: str | None = None, **kwargs):
        from langchain.chat_models import init_chat_model

        self._model = init_chat_model(model or DEFAULT_MODEL, **kwargs)

    def structured(self, schema: type[T], system: str, user: str) -> T:
        runnable = self._model.with_structured_output(schema)
        return runnable.invoke([("system", system), ("human", user)])
