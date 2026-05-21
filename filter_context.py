"""Filter context for chaining tools — per-thread memory for multi-turn chats."""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

# Per conversation thread (e.g. Streamlit IP session) — survives between ask() calls.
_thread_stores: dict[str, dict[str, FilterSpec]] = {}

_active_thread: ContextVar[str | None] = ContextVar("active_thread", default=None)

# Ephemeral store when no thread is bound (MCP one-off tool chains).
_filter_store: ContextVar[dict[str, FilterSpec] | None] = ContextVar(
    "filter_store",
    default=None,
)


class FilterSpec(BaseModel):
    """Criteria shared across filter → count / examples / distribution tools."""

    category: str | None = None
    intent: str | None = None
    intent_contains: str | None = None
    instruction_contains: str | None = None

    def normalized(self) -> FilterSpec:
        return FilterSpec(
            category=self.category.strip().upper() if self.category else None,
            intent=self.intent.strip().upper() if self.intent else None,
            intent_contains=self.intent_contains.strip().upper() if self.intent_contains else None,
            instruction_contains=(
                self.instruction_contains.strip().upper() if self.instruction_contains else None
            ),
        )

    def merge(self, **updates: str | None) -> FilterSpec:
        data = self.model_dump()
        for key, value in updates.items():
            if value is not None:
                data[key] = value
        return FilterSpec(**data).normalized()


def bind_filter_thread(thread_id: str | None) -> None:
    """Attach filter storage to a checkpoint thread (or detach with None)."""
    _active_thread.set(thread_id)


def clear_filter_thread(thread_id: str) -> None:
    """Drop all saved filters for a conversation thread."""
    _thread_stores.pop(thread_id, None)


def _parse_tool_json(content: Any) -> dict[str, Any] | None:
    text = (content if isinstance(content, str) else str(content or "")).strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        text = match.group(0)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _spec_from_filters_dict(filters: dict[str, Any]) -> FilterSpec:
    return FilterSpec(
        category=filters.get("category"),
        intent=filters.get("intent"),
        intent_contains=filters.get("intent_contains"),
        instruction_contains=filters.get("instruction_contains"),
    ).normalized()


def rehydrate_filters_from_messages(messages: list[Any]) -> None:
    """Restore filter_id entries referenced in prior tool observations."""
    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        return

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        data = _parse_tool_json(message.content)
        if not data:
            continue

        filter_id = data.get("filter_id")
        filters = data.get("filters")
        if not filter_id or not isinstance(filters, dict):
            continue

        try:
            save_filter(_spec_from_filters_dict(filters), filter_id=str(filter_id))
        except Exception:
            continue


def init_filter_context(*, reset: bool = False) -> None:
    """Ensure a filter store exists; optionally clear ephemeral (non-thread) storage."""
    thread = _active_thread.get()
    if thread:
        if reset or thread not in _thread_stores:
            _thread_stores[thread] = {}
        return
    if reset or _filter_store.get() is None:
        _filter_store.set({})


def clear_filter_context() -> None:
    """Clear filters for the active thread or ephemeral store."""
    thread = _active_thread.get()
    if thread:
        _thread_stores[thread] = {}
        return
    _filter_store.set({})


def _store() -> dict[str, FilterSpec]:
    thread = _active_thread.get()
    if thread:
        if thread not in _thread_stores:
            _thread_stores[thread] = {}
        return _thread_stores[thread]

    current = _filter_store.get()
    if current is None:
        current = {}
        _filter_store.set(current)
    return current


def save_filter(spec: FilterSpec, *, filter_id: str | None = None) -> str:
    store = _store()
    key = filter_id or uuid4().hex[:8]
    store[key] = spec.normalized()
    return key


def get_filter(filter_id: str) -> FilterSpec:
    store = _store()
    if filter_id not in store:
        raise KeyError(
            f"Unknown filter_id '{filter_id}'. Call filter_by_intent or filter_by_category first."
        )
    return store[filter_id]


def update_filter(filter_id: str, spec: FilterSpec) -> str:
    store = _store()
    store[filter_id] = spec.normalized()
    return filter_id
