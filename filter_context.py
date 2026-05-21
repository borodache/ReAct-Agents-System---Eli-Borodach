"""Filter context for chaining tools — per-thread memory with disk persistence."""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

_PROJECT_DIR = Path(__file__).resolve().parent
_FILTERS_DIR = _PROJECT_DIR / ".filters"

# Per conversation thread (Streamlit IP / CLI --session).
_thread_stores: dict[str, dict[str, FilterSpec]] = {}

# Set in ask() and refreshed from LangGraph tool config during invoke.
_bound_thread_id: str | None = None

_active_thread: ContextVar[str | None] = ContextVar("active_thread", default=None)

# Ephemeral store when no thread is bound (MCP).
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


def _resolve_thread() -> str | None:
    return _bound_thread_id or _active_thread.get()


def _filters_path(thread_id: str) -> Path:
    safe = re.sub(r"[^\w\-]", "_", thread_id.strip()) or "default"
    return _FILTERS_DIR / f"{safe}.json"


def _load_filters_disk(thread_id: str) -> dict[str, FilterSpec]:
    path = _filters_path(thread_id)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    store: dict[str, FilterSpec] = {}
    for filter_id, payload in raw.items():
        if isinstance(payload, dict):
            store[str(filter_id)] = FilterSpec.model_validate(payload).normalized()
    return store


def _persist_filters_disk(thread_id: str) -> None:
    store = _thread_stores.get(thread_id)
    if not store:
        path = _filters_path(thread_id)
        if path.is_file():
            path.unlink()
        return
    _FILTERS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {key: spec.model_dump() for key, spec in store.items()}
    _filters_path(thread_id).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def ensure_thread_from_config(config: Any) -> None:
    """Bind filter storage from LangGraph RunnableConfig (inside tool execution)."""
    if config is None:
        return
    configurable = {}
    if isinstance(config, dict):
        configurable = config.get("configurable") or {}
    else:
        configurable = getattr(config, "configurable", None) or {}
    thread_id = configurable.get("thread_id") if isinstance(configurable, dict) else None
    if thread_id:
        bind_filter_thread(str(thread_id))


def bind_filter_thread(thread_id: str | None) -> None:
    """Attach filter storage to a checkpoint thread (or detach with None)."""
    global _bound_thread_id
    _bound_thread_id = thread_id
    _active_thread.set(thread_id)
    if thread_id:
        if thread_id not in _thread_stores:
            _thread_stores[thread_id] = _load_filters_disk(thread_id)


def clear_filter_thread(thread_id: str) -> None:
    """Drop all saved filters for a conversation thread."""
    _thread_stores.pop(thread_id, None)
    path = _filters_path(thread_id)
    if path.is_file():
        path.unlink()


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


def _register_filter_from_payload(data: dict[str, Any]) -> None:
    filter_id = data.get("filter_id")
    filters = data.get("filters")
    if not filter_id or not isinstance(filters, dict):
        return
    save_filter(_spec_from_filters_dict(filters), filter_id=str(filter_id))


def rehydrate_filters_from_messages(messages: list[Any]) -> None:
    """Restore filter_id entries from prior tool observations and tool calls."""
    try:
        from langchain_core.messages import AIMessage, ToolMessage
    except ImportError:
        return

    for message in messages:
        if isinstance(message, ToolMessage):
            data = _parse_tool_json(message.content)
            if data:
                _register_filter_from_payload(data)

        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                if call.get("name") not in ("filter_by_intent", "filter_by_category"):
                    continue
                args = call.get("args") or {}
                if not isinstance(args, dict):
                    continue
                nested_id = args.get("filter_id")
                if nested_id:
                    # Parent filter must already be registered from its tool output.
                    continue
                intent = args.get("intent")
                intent_contains = args.get("intent_contains")
                category = args.get("category")
                if intent or intent_contains or category:
                    # Placeholder id will be replaced when ToolMessage arrives; skip.
                    continue


def init_filter_context(*, reset: bool = False) -> None:
    """Ensure a filter store exists; optionally clear ephemeral (non-thread) storage."""
    thread = _resolve_thread()
    if thread:
        if reset:
            _thread_stores[thread] = {}
            _persist_filters_disk(thread)
        elif thread not in _thread_stores:
            _thread_stores[thread] = _load_filters_disk(thread)
        return
    if reset or _filter_store.get() is None:
        _filter_store.set({})


def clear_filter_context() -> None:
    """Clear filters for the active thread or ephemeral store."""
    thread = _resolve_thread()
    if thread:
        _thread_stores[thread] = {}
        _persist_filters_disk(thread)
        return
    _filter_store.set({})


def _store() -> dict[str, FilterSpec]:
    thread = _resolve_thread()
    if thread:
        if thread not in _thread_stores:
            _thread_stores[thread] = _load_filters_disk(thread)
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
    thread = _resolve_thread()
    if thread:
        _persist_filters_disk(thread)
    return key


def resolve_filter_id(filter_id: str) -> str:
    """Return a valid filter_id, recovering from disk or the latest saved filter."""
    store = _store()
    if filter_id in store:
        return filter_id

    thread = _resolve_thread()
    if thread:
        _thread_stores[thread] = _load_filters_disk(thread)
        store = _thread_stores[thread]
        if filter_id in store:
            return filter_id

    if store:
        # Model often reuses an old id from chat history; use the latest saved filter.
        return next(reversed(store))

    raise KeyError(
        f"Unknown filter_id '{filter_id}'. Call filter_by_intent or filter_by_category first."
    )


def get_filter(filter_id: str) -> FilterSpec:
    resolved = resolve_filter_id(filter_id)
    return _store()[resolved]


def update_filter(filter_id: str, spec: FilterSpec) -> str:
    store = _store()
    store[filter_id] = spec.normalized()
    thread = _resolve_thread()
    if thread:
        _persist_filters_disk(thread)
    return filter_id
